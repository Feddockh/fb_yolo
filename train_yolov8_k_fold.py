#!/usr/bin/env python3
"""
K-Fold YOLOv8 trainer (Ultralytics) for a dataset laid out as:
  ROOT/
    images/
    labels/
    folds/
      fold_1/
        train.txt   # list of image paths
        val.txt
      fold_2/
        train.txt
        val.txt
      ...

What this script does:
  • For each fold directory under ROOT/folds:
      - writes a temporary data_fold.yaml that points train/val to the txt files
      - copies 'names' from BASE_DATA_YAML (recommended) so classes are correct
      - trains YOLOv8 with your chosen hyperparams
  • Reads results.csv per fold and prints a concise summary + overall mean

No CLI: edit the CONFIG section below and run:  python train_kfold_yolov8.py
"""

import os
import csv
import glob
import yaml
from pathlib import Path
from typing import Dict, List, Optional

from ultralytics import YOLO

# ========================
# CONFIG (edit these)
# ========================
pwd = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR     = os.path.join(pwd, "datasets")
ROOT_DIR        = os.path.join(DATASET_DIR, "rivendale_v6_ximea_k_fold")  # contains images/, labels/, folds/
FOLDS_DIR       = "folds"                                    # relative to ROOT_DIR
BASE_DATA_YAML  = os.path.join(ROOT_DIR, "data.yaml")        # must contain your 'names' list
MODEL_CFG       = os.path.join(pwd, "models", "yolov8l.pt")  # or a custom .pt/.yaml
PROJECT         = os.path.join(pwd, "runs/train")            # Ultralytics project dir
RUN_NAME        = "yolov8_large_rivendale_v6_ximea_k"              # base name; script appends _foldX

# Training knobs (match your style)
EPOCHS          = 100
# IMGSZ           = (1088, 1440)
IMGSZ           = (216, 409)
BATCH           = -1
PATIENCE        = 20
LR0             = 0.01
LRF             = 0.01
DROPOUT         = 0.20
AUGMENT         = True
HSV_H           = 0.0
HSV_S           = 0.0
HSV_V           = 0.0
DEGREES         = 15.0
TRANSLATE       = 0.1
SCALE           = 0.5
SHEAR           = 2.0
PERSPECTIVE     = 0.0002
FLIPUD          = 0.0
FLIPLR          = 0.5
MOSAIC          = 1.0
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


# ========================
# Helpers
# ========================

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
        )

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
