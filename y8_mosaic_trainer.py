#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import YAML
from ultralytics.nn.tasks import DetectionModel


# -----------------------------
# Geometric constants (edit!)
# -----------------------------
# Crop the raw mosaic (H0,W0) to a region that is exactly divisible by 5
CROP_TOP  = 3
CROP_LEFT = 0
CROP_H    = 1080   # height after crop
CROP_W    = 2045   # width  after crop
# After 5×5 demosaic, output size is:
OUT_H = CROP_H // 5    # 216
OUT_W = CROP_W // 5    # 409


# -----------------------------
# Utility: label path from image
# -----------------------------
def _default_label_path_from_image_path(img_path: Path) -> Path:
    # Replace /images/ with /labels/ and change extension to .txt
    p = str(img_path)
    if "/images/" in p:
        p = p.replace("/images/", "/labels/")
    else:
        # fallback: same dir
        p = str(img_path.parent / "labels" / img_path.name)
    return Path(os.path.splitext(p)[0] + ".txt")


# -----------------------------
# Geometry helpers
# -----------------------------
def clip_bbox01(xc: float, yc: float, w: float, h: float) -> Optional[Tuple[float, float, float, float]]:
    """Clip [xc,yc,w,h] normalized in [0,1] to [0,1], drop if no overlap."""
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    w = x2 - x1
    h = y2 - y1
    xc = x1 + w / 2
    yc = y1 + h / 2
    return (xc, yc, w, h)


def convert_bbox_raw_to_demosaic(xc: float, yc: float, w: float, h: float,
                                 raw_h: int, raw_w: int) -> Optional[Tuple[float, float, float, float]]:
    """
    Convert YOLO-normalized bbox on the RAW mosaic (raw_h, raw_w)
    → cropped region (CROP_TOP/CROP_LEFT, CROP_H/W)
    → demosaiced image normalized coords (OUT_H, OUT_W) with integer downsample factor 5.
    """
    # 1) raw-normalized → raw pixels
    x = xc * raw_w
    y = yc * raw_h
    bw = w * raw_w
    bh = h * raw_h

    # 2) shift to crop coordinates
    x -= CROP_LEFT
    y -= CROP_TOP

    # 3) drop boxes that lie completely outside crop in pixel space
    x1 = x - bw / 2
    y1 = y - bh / 2
    x2 = x + bw / 2
    y2 = y + bh / 2
    if x2 <= 0 or y2 <= 0 or x1 >= CROP_W or y1 >= CROP_H:
        return None

    # Clip to crop box
    x1 = np.clip(x1, 0, CROP_W)
    y1 = np.clip(y1, 0, CROP_H)
    x2 = np.clip(x2, 0, CROP_W)
    y2 = np.clip(y2, 0, CROP_H)

    # 4) downsample by 5 (integer demosaic stride)
    x1 /= 5.0; y1 /= 5.0; x2 /= 5.0; y2 /= 5.0

    # 5) normalize to OUT_W/OUT_H
    x1 /= OUT_W; x2 /= OUT_W
    y1 /= OUT_H; y2 /= OUT_H

    w_n = x2 - x1
    h_n = y2 - y1
    if w_n <= 0 or h_n <= 0:
        return None
    xc_n = x1 + w_n / 2
    yc_n = y1 + h_n / 2

    return clip_bbox01(xc_n, yc_n, w_n, h_n)


# -----------------------------
# Demosaic
# -----------------------------
def demosaic_5x5_mosaic_to_cube(cropped_gray: np.ndarray) -> np.ndarray:
    """
    Convert a 5×5 repeating mosaic (single-channel) to a [25, H/5, W/5] cube
    by striding. Band order is (r_off, c_off) flattened row-major.

    cropped_gray: (CROP_H, CROP_W), must be divisible by 5
    """
    assert cropped_gray.ndim == 2, "Expected grayscale mosaic"
    H, W = cropped_gray.shape
    assert H == CROP_H and W == CROP_W, f"Unexpected crop size: {H}x{W}"
    assert H % 5 == 0 and W % 5 == 0, "Crop dims must be divisible by 5"

    bands = []
    for r in range(5):
        for c in range(5):
            bands.append(cropped_gray[r::5, c::5])
    cube = np.stack(bands, axis=0).astype(np.float32)  # [25, H/5, W/5]
    return cube


def per_band_standardize(cube: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Per-band z-score: (x - mean) / std for each band.
    """
    m = cube.mean(axis=(1, 2), keepdims=True)
    s = cube.std(axis=(1, 2), keepdims=True)
    return (cube - m) / (s + eps)


# -----------------------------
# Dataset & collate
# -----------------------------
class YOLOMosaicDetDataset(Dataset):
    def __init__(self, list_file: Path):
        """
        list_file: file with absolute or relative image paths (one per line).
        Labels inferred via YOLO convention:
          /images/.../foo.pgm  -> /labels/.../foo.txt
          or same dir fallback.
        """
        base = list_file.parent
        self.items: List[Path] = []
        with open(list_file, "r") as f:
            for line in f:
                p = line.strip()
                if not p:
                    continue
                ip = (base / p).resolve() if not os.path.isabs(p) else Path(p)
                self.items.append(ip)
        if not self.items:
            raise RuntimeError(f"No images found from list {list_file}")

    def __len__(self): return len(self.items)

    def _read_labels(self, img_path: Path, raw_h: int, raw_w: int) -> np.ndarray:
        lp = _default_label_path_from_image_path(img_path)
        if not lp.exists():
            return np.zeros((0, 5), dtype=np.float32)
        rows = []
        with open(lp, "r") as f:
            for ln in f:
                if not ln.strip():
                    continue
                cls, xc, yc, w, h = map(float, ln.split())
                converted = convert_bbox_raw_to_demosaic(xc, yc, w, h, raw_h, raw_w)
                if converted is None:
                    continue
                xc, yc, w, h = converted
                rows.append([int(cls), xc, yc, w, h])
        return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)

    def __getitem__(self, i):
        p = self.items[i]
        raw = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(f"Could not read image {p}")
        raw_h, raw_w = raw.shape[:2]

        # crop (y1:y2, x1:x2) -> target (CROP_H, CROP_W)
        cropped = raw[CROP_TOP:CROP_TOP + CROP_H, CROP_LEFT:CROP_LEFT + CROP_W]

        # [25, OUT_H, OUT_W]
        cube = demosaic_5x5_mosaic_to_cube(cropped)
        cube = per_band_standardize(cube)

        # labels → demosaic space (normalized to OUT_W, OUT_H)
        y = self._read_labels(p, raw_h, raw_w)  # [N,5] (cls, xc, yc, w, h)
        return torch.from_numpy(cube), torch.from_numpy(y), p.stem


def collate_det(batch):
    cubes, labels, stems = zip(*batch)
    cubes = torch.stack(cubes, 0)  # [B,25,H,W]
    all_t = []
    for b, y in enumerate(labels):
        if y.numel():
            bcol = torch.full((y.shape[0], 1), b, dtype=torch.float32)
            all_t.append(torch.cat([bcol, y.float()], dim=1))
    targets = torch.cat(all_t, 0) if all_t else torch.zeros((0, 6), dtype=torch.float32)
    return cubes, targets, stems

# -----------------------------
# Ultralytics Overwrites
# -----------------------------
from ultralytics.engine.model import Model
class MosaicModel(Model):
    def __init__(self, model: Union[str, Path, "Model"] = "yolo11n.pt", task: str = None, verbose: bool = False) -> None:
        super().__init__(model=model, task=task, verbose=verbose)

from ultralytics.models import yolo
class YOLOMosaic(MosaicModel):
    def __init__(self, model: Union[str, Path] = "yolo11n.pt", task: Optional[str] = None, verbose: bool = False):
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self) -> Dict[str, Dict[str, Any]]:
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": DetectionModelMosaic,
                "trainer": Mosaic25Trainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
        }

    
class DetectionModelMosaic(DetectionModel):
    def __init__(self, cfg="yolo11n.yaml", ch=25, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)


# -----------------------------
# Custom Trainer that plugs our dataset into Ultralytics
# -----------------------------
class Mosaic25Trainer(DetectionTrainer):
    # Called by Ultralytics to construct a Dataset object
    def build_dataset(self, img_path, mode="train", batch=None, **kwargs):
        # Ultralytics passes in the string from data.yaml (train/val path)
        return YOLOMosaicDetDataset(Path(img_path))

    def get_model(self, cfg=None, weights=None, verbose=True):
        # Load cfg dict if a path was passed in
        if isinstance(cfg, (str, Path)):
            cfg = YAML().load(cfg)
        # Force input channels and Detect signature
        cfg['in_channels'] = 25
        cfg['nc'] = getattr(self.args, 'nc', cfg.get('nc', 1))
        for i, node in enumerate(cfg['head']):
            if node[2] == 'Detect':
                node[3] = [cfg['nc']]
        model = DetectionModelMosaic(cfg, ch=25, verbose=verbose)
        if weights is not None:
            model.load(weights)
        return model

    # Called to wrap the dataset into a DataLoader
    def get_dataloader(self, dataset, batch_size=16, rank=0, mode="train", shuffle=True, **kwargs):
        is_train = mode == "train"
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=self.args.workers,
            pin_memory=True,
            drop_last=is_train,
            collate_fn=collate_det,
        )

    # Be tolerant to batch format in case upstream changes
    def preprocess_batch(self, batch):
        # If Ultralytics ever hands us its own dict, just return it
        if isinstance(batch, dict) and "img" in batch:
            return batch

        # Our collate: (x, targets, stems) or rarely (x, targets, stems, extra)
        if isinstance(batch, (list, tuple)):
            x, targets, *rest = batch
            stems = rest[0] if rest else [""] * x.shape[0]
        else:
            raise TypeError(f"Unexpected batch type: {type(batch)}")

        x = x.to(self.device, non_blocking=True).float()

        if targets.numel():
            b = targets[:, 0].to(self.device)
            cls = targets[:, 1:2].to(self.device)
            bboxes = targets[:, 2:].to(self.device)
        else:
            b = torch.zeros((0,), device=self.device)
            cls = torch.zeros((0, 1), device=self.device)
            bboxes = torch.zeros((0, 4), device=self.device)

        B, _, H, W = x.shape
        return {
            "img": x,
            "batch_idx": b,
            "cls": cls,
            "bboxes": bboxes,
            "im_file": [str(s) for s in stems],
            "ori_shape": torch.tensor([[H, W]] * B, device=self.device),
        }


# -----------------------------
# Main / train entry
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="y8_25ch.yaml")
    ap.add_argument("--data_yaml", type=str, default="datasets/rivendale_v6_ximea_bandgate/data.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--imgsz_h", type=int, default=OUT_H)
    ap.add_argument("--imgsz_w", type=int, default=OUT_W)
    ap.add_argument("--project", type=str, default="runs/train")
    ap.add_argument("--name", type=str, default="y8_mosaic_experiment")
    args = ap.parse_args()

    # Build model from YAML (no pretrained)
    model = YOLOMosaic(args.model)

    # We must tell Ultralytics to use our trainer subclass.
    # Works with recent Ultralytics: pass the class object in the overrides.
    results = model.train(
        data=args.data_yaml,
        project=args.project,
        name=args.name,
        trainer=Mosaic25Trainer,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        imgsz=(args.imgsz_h, args.imgsz_w),  # rectangular OK in recent v8
        device='cpu',
        verbose=True,
        pretrained=False,
        plots=False,
        # any other overrides you like...
    )

    LOGGER.info(results)


if __name__ == "__main__":
    main()
