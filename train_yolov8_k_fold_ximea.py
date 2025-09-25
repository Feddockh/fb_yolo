import os
import csv
import glob
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import torch
import cv2
import numpy as np
from ultralytics import YOLO


# ========================
# CONFIG (edit these)
# ========================

pwd = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR     = os.path.join(pwd, "datasets")
ROOT_DIR        = os.path.join(DATASET_DIR, "rivendale_v6_ximea_k_fold_yolo")  # contains images/, labels/, folds/
FOLDS_DIR       = "folds"                                    # relative to ROOT_DIR
BASE_DATA_YAML  = os.path.join(ROOT_DIR, "data.yaml")        # must contain your 'names' list
MODEL_CFG       = os.path.join(pwd, "models", "yolov8l.pt")  # or a custom .pt/.yaml
PROJECT         = os.path.join(pwd, "runs/train")            # Ultralytics project dir
RUN_NAME        = "yolov8_large_rivendale_v6_ximea_k_demosaic_conv"              # base name; script appends _foldX

# Training knobs (match your style)
EPOCHS          = 200
# IMGSZ           = (1088, 1440)
# IMGSZ           = (216, 409) # For Ximea camera (demosaiced)
# IMGSZ           = (1080, 2045) # For Ximea camera (cropped)
IMGSZ           = (1120, 2080) # For Ximea camera (cropped + padded to multiple of 160)
BATCH           = 2
PATIENCE        = 20
LR0             = 0.01
LRF             = 0.01
DROPOUT         = 0.20
AUGMENT         = False
HSV_H           = 0.0
HSV_S           = 0.0
HSV_V           = 0.0
DEGREES         = 0.0
TRANSLATE       = 0.0
SCALE           = 0.0
SHEAR           = 0.0
PERSPECTIVE     = 0.0
FLIPUD          = 0.0
FLIPLR          = 0.0
MOSAIC          = 0.0
# MIXUP           = 0.15
# WEIGHT_DECAY    = 0.0005
WARMUP_EPOCHS   = 3.0
COS_LR          = True

# Inference/val-time NMS knobs (affect validation metrics, not training loss)
CONF_THRESH     = 0.1
IOU_NMS         = 0.3
MAX_DET         = 300
AGNOSTIC_NMS    = False

DEVICE          = None  # e.g. "0" or "0,1" or "cpu"; None = auto
WORKERS         = 8
EXIST_OK        = True
VERBOSE         = True

ADD_DEMOSAIC_CONV = True  # prepend Conv2d(5x5,s=5) to model for Ximea mosaiced input
RECT = True # rectangular training (avoid distortion when using non-square imgsz)


# ========================
# Helpers
# ========================

def load_names_from_base_yaml(path: str) -> List[str]:
    import yaml
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    names = d.get("names")
    if names is None:
        raise ValueError(f"'names' not found in {path}. Provide class names.")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    return list(names)

def build_data_yaml(train_txt: str, val_txt: str, names: List[str], base_path: Optional[str] = None) -> Dict:
    """
    NOTE: 'path' here is not relied upon by YOLO for lines *inside* list files.
    We still include it for completeness, but we ensure list files have absolute lines.
    """
    d = {
        "train": train_txt,
        "val": val_txt,
        "names": names,
    }
    if base_path is not None:
        d["path"] = base_path  # harmless, useful if you ever point to dirs instead of TXT
    return d

def _rewrite_list_to_absolute(src_txt: Path, dataset_root: Path, dst_txt: Path) -> None:
    """
    Read each line from src_txt (e.g., 'images/foo.png') and write an ABSOLUTE path
    rooted at dataset_root. Skips empty/comment lines.
    """
    lines_out = []
    with open(src_txt, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line)
            if not p.is_absolute():
                p = dataset_root / line
            lines_out.append(str(p.resolve()))
    dst_txt.write_text("\n".join(lines_out) + "\n")

def _quick_sanity(img_list_txt: Path) -> None:
    """
    Optional: print a couple of lines to verify the absolute rewrite worked.
    """
    try:
        with open(img_list_txt, "r") as f:
            sample = [next(f).strip() for _ in range(3)]
        print(f"Sanity sample from {img_list_txt.name}:")
        for s in sample:
            print("  ", s)
    except StopIteration:
        print(f"{img_list_txt.name} appears empty.")

def discover_folds(root: str, folds_rel: str) -> List[Path]:
    folds_root = Path(root) / folds_rel
    if not folds_root.exists():
        raise FileNotFoundError(f"Folds directory not found: {folds_root}")
    # Accept any subdir with train.txt and val.txt
    candidates = []
    for d in sorted(folds_root.iterdir()):
        if d.is_dir() and (d / "train.txt").exists() and (d / "val.txt").exists():
            candidates.append(d)
    if not candidates:
        raise RuntimeError(f"No folds found under {folds_root} (expect fold_x/train.txt & val.txt).")
    return candidates

def read_last_row_csv(csv_path: Path) -> Optional[Dict[str, str]]:
    if not csv_path.exists():
        return None
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        last = None
        for row in reader:
            last = row
    return last

def add_demosaic_conv(model: YOLO) -> None:
    import torch
    import torch.nn as nn
    from ultralytics.nn.modules.conv import Conv  # <-- use Ultralytics Conv

    det = model.model  # DetectionModel
    assert hasattr(det, "model"), "Expected DetectionModel with .model"

    # Ultralytics Conv = Conv2d + BN + SiLU; k=5, s=5, p=0 (top-left anchor)
    front = Conv(c1=3, c2=3, k=5, s=5, p=0, act=True)
    # Add the graph attribute Ultralytics expects
    setattr(front, "f", -1)      # from = previous
    setattr(front, "i", -1)      # layer index (optional)
    setattr(front, "type", type(front).__name__)

    # Prepend INSIDE DetectionModel (keep DetectionModel wrapper intact)
    det.model = nn.Sequential(front, det.model)

    # Update effective stride (relative to original image)
    if hasattr(det, "stride"):
        s = det.stride
        factor = 5
        if isinstance(s, torch.Tensor):
            det.stride = s * factor
        elif isinstance(s, (list, tuple)):
            det.stride = type(s)(int(x) * factor for x in s)
        else:
            det.stride = int(s) * factor

    # Prevent trainer/predictor from trying to reload checkpoint shapes
    model.ckpt = None
    if hasattr(model, "overrides"):
        model.overrides["pretrained"] = False

    print("[kfold] Prepended Ultralytics Conv(k=5,s=5); stride×5; disabled reload")



def _effective_max_stride(model: YOLO) -> int:
    s = getattr(model.model, "stride", 32)
    print(f"[kfold] Model stride attribute: {s}")
    try:
        # torch tensor or sequence -> take max
        if hasattr(s, "max"):
            return int(s.max())
        if isinstance(s, (list, tuple)):
            return int(max(s))
        return int(s)
    except Exception:
        return 32
    
def check_imgsz_vs_stride(imgsz, model: YOLO) -> None:
    max_stride = _effective_max_stride(model)
    h, w = imgsz
    if h % max_stride != 0 or w % max_stride != 0:
        print(f"WARNING: imgsz {imgsz} not multiple of model max stride {max_stride}. "
              "Consider adjusting imgsz or padding to multiple of stride.")
        
def _to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
    """t: CHW or BCHW float tensor. Return HxWx3 uint8 RGB."""
    if t.ndim == 4:
        t = t[0]
    if t.shape[0] < 3:
        t = t.repeat(3, 1, 1)
    t = t[:3].detach().cpu().float()
    t_min, t_max = float(t.min()), float(t.max())
    if t_max > 1.5 or t_min < 0.0:
        t = (t - t_min) / max(1e-6, (t_max - t_min))
    img = (t.permute(1, 2, 0).clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    return img

def _prep_one_image_top_left(img_path: str, imgsz: int, pad_val: int = 114) -> np.ndarray:
    """
    Load BGR with cv2, convert to RGB, resize keeping aspect to fit in (imgsz, imgsz),
    then pad on bottom/right with pad_val. Top-left anchor.
    Returns HxWx3 uint8 RGB canvas of size imgsz x imgsz.
    """
    im0 = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if im0 is None:
        raise FileNotFoundError(img_path)
    im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
    h0, w0 = im0.shape[:2]
    scale = min(imgsz / h0, imgsz / w0)
    nh, nw = int(round(h0 * scale)), int(round(w0 * scale))
    imr = cv2.resize(im0, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), pad_val, dtype=np.uint8)
    canvas[:nh, :nw] = imr  # top-left anchor
    return canvas

def visualize_front_conv_io(model: YOLO, image_paths, out_dir: Path, imgsz, device=None, max_images: int = 8):
    """
    Run a plain forward through model.model (DetectionModel) so hooks fire on the
    prepended demosaic conv. Saves *_before_front.png and *_after_front.png.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find the prepended front conv at DetectionModel.model[0]
    front = None
    if hasattr(model, "model") and hasattr(model.model, "model"):
        inner = model.model.model
        if isinstance(inner, torch.nn.Sequential) and len(inner) >= 1:
            front = inner[0]
    if front is None:
        print("[viz] Could not locate the front conv at model.model.model[0].")
        return

    # Ensure imgsz is an int
    imgsz_int = int(max(imgsz)) if isinstance(imgsz, (tuple, list)) else int(imgsz)

    # Resolve device
    if device is None or device == "auto":
        try:
            device = next(model.model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        device = torch.device(device)

    captured_in, captured_out = [], []

    def pre_hook(module, inputs):
        captured_in.clear()
        captured_in.append(inputs[0])  # BCHW

    def fwd_hook(module, inputs, output):
        captured_out.clear()
        captured_out.append(output)    # BCHW

    h1 = front.register_forward_pre_hook(pre_hook)
    h2 = front.register_forward_hook(fwd_hook)

    # Eval, no grad
    was_training = model.model.training
    model.model.eval()
    torch.set_grad_enabled(False)

    try:
        count = 0
        for p in image_paths:
            if count >= max_images:
                break

            # --- preprocess (letterbox top-left to imgsz x imgsz) ---
            canvas = _prep_one_image_top_left(str(p), imgsz_int, pad_val=114)  # HWC RGB uint8
            x = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0      # CHW float in [0,1]
            x = x.unsqueeze(0).to(device, non_blocking=True)                    # BCHW

            # --- forward (direct, no AutoBackend, no NMS) ---
            _ = model.model(x)

            if not captured_in or not captured_out:
                print(f"[viz] No tensors captured for {p}")
                continue

            img_in  = _to_uint8_rgb(captured_in[0])
            img_out = _to_uint8_rgb(captured_out[0])

            stem = Path(p).stem
            cv2.imwrite(str(out_dir / f"{stem}_before_front.png"), cv2.cvtColor(img_in,  cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_dir / f"{stem}_after_front.png"),  cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR))
            count += 1

        print(f"[viz] Saved {count*2} images (before/after) to {out_dir}")
    finally:
        h1.remove()
        h2.remove()
        model.model.train(was_training)
        torch.set_grad_enabled(True)


# ========================
# Main training loop
# ========================

def train_kfold_yolov8():
    root = Path(ROOT_DIR).resolve()
    folds = discover_folds(str(root), FOLDS_DIR)
    names = load_names_from_base_yaml(BASE_DATA_YAML)

    summary_rows = []
    print(f"Found {len(folds)} folds in {root / FOLDS_DIR}\n")

    for idx, fold_dir in enumerate(folds, start=1):
        train_txt_rel = (fold_dir / "train.txt").resolve()
        val_txt_rel   = (fold_dir / "val.txt").resolve()

        # Create absolute-path list files per fold
        train_txt_abs = fold_dir / "train_abs.txt"
        val_txt_abs   = fold_dir / "val_abs.txt"
        _rewrite_list_to_absolute(train_txt_rel, root, train_txt_abs)
        _rewrite_list_to_absolute(val_txt_rel, root, val_txt_abs)

        # (optional) quick peek
        _quick_sanity(train_txt_abs)

        # Create per-fold data.yaml
        data_yaml_path = fold_dir / "data_fold.yaml"
        data_dict = build_data_yaml(str(train_txt_abs), str(val_txt_abs), names, base_path=str(root))
        with open(data_yaml_path, "w") as f:
            yaml.safe_dump(data_dict, f, sort_keys=False)

        run_name = f"{RUN_NAME}_fold{idx}"
        print(f"=== Training {run_name} ===")
        print(f"train list: {train_txt_abs}")
        print(f"val list:   {val_txt_abs}")

        model = YOLO(MODEL_CFG)

        if ADD_DEMOSAIC_CONV:
            add_demosaic_conv(model)
            check_imgsz_vs_stride(IMGSZ, model)

        # pick a handful of training image paths to visualize
        with open(train_txt_abs, "r") as f:
            train_paths = [ln.strip() for ln in f.readlines() if ln.strip()]
        sample_paths = train_paths[:6]  # visualize first 6

        # visualize before/after the front conv
        viz_dir = Path(PROJECT) / run_name / "front_conv_viz"
        visualize_front_conv_io(
            model=model,
            image_paths=sample_paths,
            out_dir=viz_dir,
            imgsz=IMGSZ,          # your current training imgsz (tuple is fine; function will coerce)
            device=DEVICE,
            max_images=6,
        )

        # Train
        model.train(
            data=str(data_yaml_path),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            patience=PATIENCE,
            lr0=LR0,
            lrf=LRF,
            dropout=DROPOUT,
            augment=AUGMENT,
            hsv_h=HSV_H, hsv_s=HSV_S, hsv_v=HSV_V,
            degrees=DEGREES,
            translate=TRANSLATE,
            scale=SCALE,
            shear=SHEAR,
            perspective=PERSPECTIVE,
            flipud=FLIPUD,
            fliplr=FLIPLR,
            mosaic=MOSAIC,
            # mixup=MIXUP,
            # weight_decay=WEIGHT_DECAY,
            warmup_epochs=WARMUP_EPOCHS,
            cos_lr=COS_LR,
            project=PROJECT,
            name=run_name,
            exist_ok=EXIST_OK,
            device=DEVICE,
            workers=WORKERS,
            verbose=VERBOSE,

            # NMS knobs for val/inference during training
            conf=CONF_THRESH,
            iou=IOU_NMS,
            max_det=MAX_DET,
            agnostic_nms=AGNOSTIC_NMS,

            rect=RECT,
            pretrained=False if ADD_DEMOSAIC_CONV else True,  # don't try to load if we changed the model
        )

        # Print out the 

        # Grab metrics from results.csv (last row ~ final epoch / best epoch depending on settings)
        run_dir = Path(PROJECT) / run_name
        csv_path = run_dir / "results.csv"
        last = read_last_row_csv(csv_path)
        row_out = {"fold": idx, "run_dir": str(run_dir)}
        if last:
            # Common columns (Ultralytics 8.x). If a key is missing, ignore.
            for key in [
                "metrics/mAP50-95(B)", "metrics/mAP50(B)", "metrics/precision(B)",
                "metrics/recall(B)", "fitness"
            ]:
                if key in last:
                    row_out[key] = last[key]
        summary_rows.append(row_out)

        print(f"Finished fold {idx}. Results CSV: {csv_path}\n")

    # Print a compact summary and compute means where possible
    print("\n==== K-Fold Summary ====")
    headers = ["fold", "metrics/mAP50-95(B)", "metrics/mAP50(B)", "metrics/precision(B)", "metrics/recall(B)", "run_dir"]
    print("\t".join(h for h in headers))
    m50_95_vals, m50_vals = [], []
    for r in summary_rows:
        line = [
            str(r.get("fold", "")),
            str(r.get("metrics/mAP50-95(B)", "")),
            str(r.get("metrics/mAP50(B)", "")),
            str(r.get("metrics/precision(B)", "")),
            str(r.get("metrics/recall(B)", "")),
            r.get("run_dir", ""),
        ]
        print("\t".join(line))
        # collect for means
        try:
            m50_95_vals.append(float(r.get("metrics/mAP50-95(B)", "")))
        except Exception:
            pass
        try:
            m50_vals.append(float(r.get("metrics/mAP50(B)", "")))
        except Exception:
            pass

    if m50_95_vals:
        mean_5095 = sum(m50_95_vals) / len(m50_95_vals)
        print(f"\nMean mAP@0.50:0.95 across folds: {mean_5095:.4f}")
    if m50_vals:
        mean_50 = sum(m50_vals) / len(m50_vals)
        print(f"Mean mAP@0.50 across folds:      {mean_50:.4f}")


if __name__ == "__main__":
    os.chdir(ROOT_DIR)

    train_kfold_yolov8()
