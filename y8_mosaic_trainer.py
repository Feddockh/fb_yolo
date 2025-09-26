#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import json

import cv2
from cv2 import imread
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import YAML
from ultralytics.nn.tasks import DetectionModel

from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import RANK, colorstr, callbacks, TQDM, emojis
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first, smart_inference_mode, select_device
from ultralytics.data import build_yolo_dataset
from ultralytics.data.utils import IMG_FORMATS, PIN_MEMORY, VID_FORMATS
from torch.utils.data import dataloader, distributed
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.engine.model import Model
from ultralytics.models import yolo

from copy import copy, deepcopy
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results


# -----------------------------
# Geometric constants (edit!)
# -----------------------------
UNCROPPED_H = 1088  # original image height
UNCROPPED_W = 2048  # original image width
# Crop the raw mosaic (H0,W0) to a region that is exactly divisible by 5
CROP_TOP  = 3
CROP_LEFT = 0
CROP_H    = 1080   # height after crop
CROP_W    = 2045   # width  after crop
# After 5×5 demosaic, output size is:
OUT_H = CROP_H // 5    # 216
OUT_W = CROP_W // 5    # 409

BANDWIDTHS = np.array([
    [886, 896, 877, 867, 951],
    [793, 806, 782, 769, 675],
    [743, 757, 730, 715, 690],
    [926, 933, 918, 910, 946],
    [846, 857, 836, 824, 941],
])

USING_BANDS = BANDWIDTHS.ravel().tolist()  # Use all bands by default


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

    bbox = clip_bbox01(xc_n, yc_n, w_n, h_n)
    return np.array(bbox, dtype=np.float32)


# -----------------------------
# Demosaic Functionality
# -----------------------------
def demosaic_5x5_mosaic_to_cube(cropped_gray: np.ndarray) -> np.ndarray:
    """
    Convert a 5×5 repeating mosaic (single-channel) to a [n, H/5, W/5] cube
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
    
    # Find the indexes of the requested bands
    index_map = {band: idx for idx, band in enumerate(BANDWIDTHS.ravel().tolist())}
    matching_indexes = [index_map[b] for b in USING_BANDS if b in index_map]

    # Re-assemble cube with only the requested bands
    cube = np.array([bands[i] for i in matching_indexes])  # [n, H/5, W/5]

    return cube

# Don't use this, blacks out the image
# def per_band_standardize(cube: np.ndarray, eps: float = 1e-6) -> np.ndarray:
#     """
#     Per-band z-score: (x - mean) / std for each band.
#     """
#     m = cube.mean(axis=(1, 2), keepdims=True)
#     s = cube.std(axis=(1, 2), keepdims=True)
#     return (cube - m) / (s + eps)

def demosaic_im_read(file_path: str) -> np.ndarray:
    """
    Read a raw mosaic image from file, crop, demosaic to cube, and per-band standardize.

    Returns:
        (np.ndarray): [n, OUT_H, OUT_W] cube of float32
    """
    raw = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Load raw mosaic
    if raw is None:
        raise FileNotFoundError(f"Could not read image {file_path}")
    raw_h, raw_w = raw.shape[:2]
    cropped = raw[CROP_TOP:CROP_TOP + CROP_H, CROP_LEFT:CROP_LEFT + CROP_W]
    cube = demosaic_5x5_mosaic_to_cube(cropped)  # [n, 216, 409]
    # cube = per_band_standardize(cube)  # Per-band z-score normalization
    cube = cube.transpose(1, 2, 0)  # [H,W,n] for augmentation compatibility
    return cube.astype(np.float32)


# -----------------------------
# Dataset & collate
# -----------------------------
class YOLOMosaicDetDataset(YOLODataset):
    def __init__(self, *args, data: Optional[Dict] = None, task: str = "detect", **kwargs):
        super().__init__(*args, data=data, task=task, **kwargs)

    def load_image(self, i, rect_mode = True):
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = demosaic_im_read(f)  # 25 bands
            else:  # read image
                im = demosaic_im_read(f)  # 25 bands
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            if im.ndim == 2:
                im = im[..., None]

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]
    
    def get_image_and_label(self, index: int) -> Dict[str, Any]:
        label = deepcopy(self.labels[index])

        # Adjust the labels to demosaic space
        bboxes = label["bboxes"]
        for i in range(len(bboxes)):
            box = bboxes[i]
            converted_box = convert_bbox_raw_to_demosaic(box[0], box[1], box[2], box[3],
                                                        UNCROPPED_H, UNCROPPED_W)
            bboxes[i] = converted_box if converted_box is not None else box
        
        label.pop("shape", None)
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

# -----------------------------
# Ultralytics Overwrites
# -----------------------------
class MosaicModel(Model):
    def __init__(self, model: Union[str, Path, "Model"] = "yolov8l.yaml", task: str = None, verbose: bool = False) -> None:
        super().__init__(model=model, task=task, verbose=verbose)

class YOLOMosaic(MosaicModel):
    def __init__(self, model: Union[str, Path] = "yolov8l.yaml", task: Optional[str] = None, verbose: bool = False):
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
    def __init__(self, cfg="yolov8l.yaml", ch=len(USING_BANDS), nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

# -----------------------------
# Custom Trainer that plugs our dataset into Ultralytics
# -----------------------------
class Mosaic25Trainer(DetectionTrainer):
    # Called by Ultralytics to construct a Dataset object
    def build_dataset(self, img_path: str, mode: str = "train", batch: Optional[int] = None):
        # Ultralytics passes in the string from data.yaml (train/val path)
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        cfg = self.args
        
        return YOLOMosaicDetDataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=mode == "train",  # augmentation
            hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
            rect=cfg.rect or False,  # rectangular batches
            cache=cfg.cache or None,
            single_cls=cfg.single_cls or False,
            stride=int(gs),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=cfg.task,
            classes=cfg.classes,
            data=self.data,
            fraction=cfg.fraction if mode == "train" else 1.0,
        )
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        # Load cfg dict if a path was passed in
        if isinstance(cfg, (str, Path)):
            cfg = YAML().load(cfg)
        # Force input channels and Detect signature
        cfg['in_channels'] = len(USING_BANDS)  # 25 input channels
        cfg['nc'] = getattr(self.args, 'nc', cfg.get('nc', 1))
        for i, node in enumerate(cfg['head']):
            if node[2] == 'Detect':
                node[3] = [cfg['nc']]
        model = DetectionModelMosaic(cfg, ch=cfg['in_channels'], verbose=verbose)
        if weights is not None:
            model.load(weights)
        return model
    
    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        validator = Mosaic25Validator(
            self.test_loader, 
            save_dir=self.save_dir, 
            args=copy(self.args), 
            _callbacks=self.callbacks
        )
        # Ensure validator has the correct data config with 25 channels
        validator.data = copy(self.data)
        validator.data["channels"] = len(USING_BANDS)  # Override channel count for warmup
        return validator
    
    def plot_training_samples(self, batch: Dict[str, Any], ni: int) -> None:
        """
        Plot training samples with their annotations.

        Args:
            batch (Dict[str, Any]): Dictionary containing batch data.
            ni (int): Number of iterations.
        """
        # The batch['img'] is already in [B,25,H,W] format
        # Want to drop to [B,1,H,W] for plotting
        batch['img'] = batch['img'][:, :1, :, :]  # keep only first band for visualization

        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )
    
# -----------------------------
# Custom Validator that overwrites Ultralytics
# -----------------------------
class Mosaic25Validator(DetectionValidator):
    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        super().__init__(dataloader=dataloader, save_dir=save_dir, args=args, _callbacks=_callbacks)

    def plot_val_samples(self, batch: Dict[str, Any], ni: int) -> None:
        """
        Plot validation image samples.

        Args:
            batch (Dict[str, Any]): Batch containing images and annotations.
            ni (int): Batch index.
        """
        # The batch['img'] is already in [B,25,H,W] format
        # Want to drop to [B,1,H,W] for plotting
        batch['img'] = batch['img'][:, :1, :, :]  # keep only first band for visualization

        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def build_dataset(self, img_path, mode = "val", batch = None):
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        cfg = self.args
        
        return YOLOMosaicDetDataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=mode == "train",  # augmentation
            hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
            rect=cfg.rect or False,  # rectangular batches
            cache=cfg.cache or None,
            single_cls=cfg.single_cls or False,
            stride=int(gs),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=cfg.task,
            classes=cfg.classes,
            data=self.data,
            fraction=cfg.fraction if mode == "train" else 1.0,
        )
    
    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """
        Execute validation process, running inference on dataloader and computing performance metrics.

        Args:
            trainer (object, optional): Trainer object that contains the model to validate.
            model (nn.Module, optional): Model to validate if not using a trainer.

        Returns:
            (dict): Dictionary containing validation statistics.
        """
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            # Force FP16 val during training
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            if str(self.args.model).endswith(".yaml") and model is None:
                LOGGER.warning("validating an untrained model YAML will result in 0 mAP.")
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                weights=model or self.args.model,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            elif not (pt or jit or getattr(model, "dynamic", False)):
                self.args.batch = model.metadata.get("batch", 1)  # export.py models default to batch-size 1
                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

            if str(self.args.data).rsplit(".", 1)[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not (pt or (getattr(model, "dynamic", False) and not model.imx)):
                self.args.rect = False
            self.stride = model.stride  # used in get_dataloader() for padding
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            self.data['channels'] = len(USING_BANDS)  # Override channel count for warmup

            model.eval()
            model.warmup(imgsz=(1 if pt else self.args.batch, self.data["channels"], imgsz, imgsz))  # warmup

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference
            with dt[1]:
                preds = model(batch["img"], augment=augment)

            # Loss
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]

            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")
        stats = self.get_stats()
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            LOGGER.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w", encoding="utf-8") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats
    


# -----------------------------
# Main / train entry
# -----------------------------
def main():
    global USING_BANDS

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="y8_25chl.yaml")
    ap.add_argument("--data_yaml", type=str, default="datasets/rivendale_v6_ximea_bandgate/data.yaml")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--imgsz_h", type=int, default=OUT_H)
    ap.add_argument("--imgsz_w", type=int, default=OUT_W)
    ap.add_argument("--project", type=str, default="runs/train")
    ap.add_argument("--name", type=str, default="yolov8_n_band_input_experiment")
    ap.add_argument("--bands", type=str,
        default=",".join(map(str, BANDWIDTHS.ravel().tolist())),
        help="Comma-separated list of band ids (row-major 5x5) or 'auto'/'default' to use BANDWIDTHS",
    )
    args = ap.parse_args()

    # Parse bands into a Python list of ints (or use default BANDWIDTHS)
    if isinstance(args.bands, str):
        s = args.bands.strip()
        if s.lower() in ("auto", "default"):
            USING_BANDS = BANDWIDTHS.ravel().tolist()
        else:
            USING_BANDS = [int(x) for x in s.split(",") if x.strip()]

    LOGGER.info(f"Using bands: {USING_BANDS} (total {len(USING_BANDS)})")
    args.name = f"yolov8_{len(USING_BANDS)}_band_input_experiment"

    # Ultralytics datasets often expect a single imgsz scalar internally;
    # keep imgsz_h/imgsz_w for the trainer override but provide imgsz as max for internal use.
    args.imgsz = max(args.imgsz_h, args.imgsz_w)

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
        imgsz=(args.imgsz_h, args.imgsz_w),
        device='cuda:0',
        verbose=True,
        pretrained=False,
        plots=True,
        augment=False,
        hsv_h       = 0.0,    # Hue augmentation
        hsv_s       = 0.0,    # Saturation augmentation  
        hsv_v       = 0.0,    # Value augmentation
        degrees     = 0.0,    # Random rotation
        translate   = 0.1,    # Random translation
        scale       = 0.5,    # Random scaling
        shear       = 0.0,    # Random shear
        perspective = 0.0,    # Perspective transformation
        flipud      = 0.0,    # Vertical flip probability
        fliplr      = 0.5,    # Horizontal flip probability
        mosaic      = 0.0,    # Keep mosaic augmentation
        mixup       = 0.0,    # Add mixup augmentation
        copy_paste  = 0.0,    # Add copy-paste augmentation
        rect=True,
        # any other overrides you like...
    )

    # LOGGER.info(results)


if __name__ == "__main__":
    main()
