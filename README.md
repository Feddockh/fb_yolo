# Fire Blight Detection with Multispectral YOLO

Training, inference, and analysis tools for detecting fire blight (*Erwinia amylovora*) symptoms in apple orchards using YOLOv8/v11 object detection. Supports both standard RGB images and 25-band hyperspectral input from Ximea NIR mosaic cameras, with optional learned spectral band selection (BandGate).

**Detection classes:** Shepherd's Crook, Canker

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Installation](#installation)
3. [Dataset Setup](#dataset-setup)
4. [Spectral Band Information](#spectral-band-information)
5. [Model Architectures](#model-architectures)
6. [Training](#training)
   - [Standard RGB (YOLOv8 / YOLOv11)](#standard-rgb-yolov8--yolov11)
   - [K-Fold Cross-Validation](#k-fold-cross-validation)
   - [25-Band Mosaic Training](#25-band-mosaic-training)
   - [Bandgated Spectral Selection](#bandgated-spectral-selection)
   - [Instance Segmentation](#instance-segmentation)
7. [Inference](#inference)
8. [Validation](#validation)
9. [Band Importance Analysis](#band-importance-analysis)
10. [ONNX Export](#onnx-export)
11. [Helper Scripts](#helper-scripts)

---

## Repository Structure

```
fb_yolo/
├── train_yolov8.py               # Standard YOLOv8 training (RGB)
├── train_yolov11.py              # YOLOv11 training (RGB)
├── train_yolov8_k_fold.py        # K-fold cross-validation (RGB)
├── train_yolov8_k_fold_ximea.py  # K-fold cross-validation (25-band Ximea)
├── train_yolo26_seg.py           # Instance segmentation (YOLO26)
├── train_bandgated_yololite.py   # Soft BandGate training (25-band)
├── train_bandgated_hard_yololite.py  # Hard (binary) BandGate training
├── train_model.sh                # Example training commands
├── y8_mosaic_trainer.py          # Custom trainer for 25-channel input
├── y8_mosaic_val.py              # Validation for 25-channel models
├── infer.py                      # Single-image inference with GT overlay
├── infer_batch.py                # Batch inference over a directory
├── infer_streamlit.py            # Streamlit web UI for inference
├── val.py                        # Validation (mAP, PR curves)
├── custom_detection_validator.py # Validation with configurable IoU range
├── model_analyze.py              # Spectral band importance from weights
├── y8_25ch{n,s,m,l}.yaml         # YOLOv8 arch configs for 25-channel input
├── ONNX_export.md                # ONNX export instructions
├── requirements.txt
├── datasets/                     # Symlink or local datasets directory
├── models/                       # Pretrained .pt weights
├── band_analysis/                # Band importance outputs (CSV, JSON, plots)
├── helpers/                      # Preprocessing & dataset utility scripts
├── huggingface_roboflow/         # HuggingFace / Roboflow deployment tools
└── runs/                         # Ultralytics training + inference outputs
```

---

## Installation

**Requirements:** Python 3.10+, CUDA-capable GPU recommended.

```bash
pip install -r requirements.txt
```

Key dependencies: `torch==2.5.1`, `torchvision==0.20.1`, `ultralytics==8.3.169`, `opencv-python`, `scikit-image`, `pandas`, `streamlit`.

---

## Dataset Setup

Datasets are expected at `datasets/` relative to the project root. If your data lives elsewhere, create a symlink:

```bash
ln -s /path/to/your/datasets datasets
```

### Dataset naming conventions

| Dataset | Description |
|---|---|
| `rivendale_v5`, `rivendale_v6` | Main RGB orchard dataset (Rivendale farm) |
| `rivendale_v6_ximea_k_fold` | Raw Ximea mosaic (.pgm) with k-fold splits |
| `rivendale_v6_ximea_bandgate` | Same as above, organized for BandGate training |
| `lab_fb_images`, `lab_fb_images_v2` | Lab-collected fire blight samples (RGB) |
| `penn_state_rivendale_seg` | Penn State + Rivendale combined segmentation dataset |
| `penn_state_seg` | Penn State segmentation only |
| `gazebo_fb_images`, `pybullet_fb_images` | Synthetic images (Gazebo / PyBullet simulation) |

### Expected folder structure

```
dataset_root/
    images/
        train/          # *.png (RGB) or *.pgm (raw mosaic)
        val/
    labels/
        train/          # YOLO format: <cls> <xc> <yc> <w> <h>  (normalized)
        val/
    data.yaml           # 'path', 'train', 'val', 'names', 'nc'
```

### K-fold structure

```
dataset_root/
    images/             # all images (flat)
    labels/             # all labels (flat)
    folds/
        fold_1/
            train.txt   # absolute paths to training images
            val.txt
        fold_2/
            ...
    data.yaml
```

Run `helpers/make_val.py` to do an 80/20 train/val split from an existing flat dataset.

---

## Spectral Band Information

The Ximea camera captures a **1088×2048 single-channel RAW mosaic** using a 5×5 repeating Bayer-like filter pattern, encoding **25 spectral bands** in the visible–NIR range (675–951 nm).

### Band wavelengths (nm) — 5×5 mosaic pattern

| | col 0 | col 1 | col 2 | col 3 | col 4 |
|---|---|---|---|---|---|
| **row 0** | 886 | 896 | 877 | 867 | 951 |
| **row 1** | 793 | 806 | 782 | 769 | 675 |
| **row 2** | 743 | 757 | 730 | 715 | 690 |
| **row 3** | 926 | 933 | 918 | 910 | 946 |
| **row 4** | 846 | 857 | 836 | 824 | 941 |

### Raw → demosaiced coordinate transform

| Step | Operation | Result size |
|---|---|---|
| Raw mosaic | Input | 1088 × 2048 |
| Crop | Remove top 3 rows, right 3 columns | 1080 × 2045 |
| Demosaic | 5×5 stride-5 sampling into 25 channels | 216 × 409 × 25 |

YOLO bounding box coordinates must be re-scaled accordingly (handled automatically by `y8_mosaic_trainer.py`, `train_yolov8_k_fold_ximea.py`, and `helpers/batch_demosaic.py`).

Band importance analysis (see [Band Importance Analysis](#band-importance-analysis)) shows that **NIR bands at 836, 824, and 926 nm** carry the most discriminative information for fire blight detection.

---

## Model Architectures

### 25-channel YOLOv8 configs (`y8_25ch*.yaml`)

Modified YOLOv8 architecture definitions that replace the standard 3-channel RGB input with 25 spectral bands. Four sizes are available:

| Config | Size | First conv |
|---|---|---|
| `y8_25chn.yaml` | Nano | 25 → 16 |
| `y8_25chs.yaml` | Small | 25 → 32 |
| `y8_25chm.yaml` | Medium | 25 → 32 |
| `y8_25chl.yaml` | Large | 25 → 64 |

Pass these as the `model` argument instead of a pretrained `.pt` file when starting from scratch.

### BandGate module

Implemented in `train_bandgated_yololite.py` and `train_bandgated_hard_yololite.py`. A learnable per-band weighting layer is prepended to a lightweight YOLO-style backbone:

- **Soft BandGate** — continuous importance weights (α parameters) trained jointly with the detector. Band rankings are printed after each epoch.
- **Hard BandGate** — binary band selection; after training, exactly *k* bands are active, producing an interpretable sparse selection.

### DemosaicConv (on-the-fly)

`y8_mosaic_trainer.py` implements an optional `DemosaicConv` prepend layer that accepts raw 1088×2048 mosaic images directly and performs demosaicing as part of the forward pass, avoiding the need to pre-process images offline.

---

## Training

All training scripts write outputs under `runs/train/<name>/weights/best.pt`.

### Standard RGB (YOLOv8 / YOLOv11)

Edit the `MODEL`, `DATASET`, and `name` constants at the top of each script, then run:

```bash
python train_yolov8.py
python train_yolov11.py
```

Key defaults: `imgsz=(1088, 1440)`, 200 epochs, auto batch size, cosine LR off, standard YOLO augmentations.

### K-Fold Cross-Validation

Trains one model per fold sequentially and saves each under `runs/train/<RUN_NAME>_fold<N>/`.

**RGB k-fold:**
```bash
python train_yolov8_k_fold.py
```

**25-band Ximea k-fold** (raw .pgm input, on-the-fly demosaic):
```bash
python train_yolov8_k_fold_ximea.py
```

Edit `ROOT_DIR`, `MODEL_CFG`, `RUN_NAME`, and training knobs (`EPOCHS`, `IMGSZ`, `BATCH`, etc.) at the top of each script.

### 25-Band Mosaic Training

Uses `y8_mosaic_trainer.py` which subclasses the Ultralytics `DetectionTrainer` to inject on-the-fly demosaicing and optional band subset selection.

```bash
# Train with all 25 bands (YOLOv8m)
python y8_mosaic_trainer.py --model yolov8m.pt --name yolov8m_25_band_input_experiment

# Train with only 3 selected bands (836, 824, 926 nm)
python y8_mosaic_trainer.py --model yolov8n.pt \
    --name yolov8n_836824926_band_input_experiment \
    --bands "836, 824, 926"
```

Pass a `y8_25ch*.yaml` config as `--model` to initialize a fresh 25-channel architecture rather than fine-tuning from a 3-channel pretrained weight.

### Bandgated Spectral Selection

Trains a lightweight custom detector from scratch while simultaneously learning which spectral bands are most informative.

**Soft gating** (continuous weights, all bands always active during training):
```bash
python train_bandgated_yololite.py \
    --data datasets/rivendale_v6_ximea_bandgate/data.yaml \
    --epochs 100 \
    --batch 16 \
    --lr 0.002 \
    --l1 0.001 \
    --workers 4
```

**Hard gating** (binary selection, sparsity enforced):
```bash
python train_bandgated_hard_yololite.py \
    --data datasets/rivendale_v6_ximea_bandgate/data.yaml \
    --epochs 100 \
    --batch 16
```

`train_model.sh` contains additional example commands for multi-run band selection experiments.

Runs are saved under `runs/bandgate/<timestamp>/`.

### Instance Segmentation

```bash
python train_yolo26_seg.py
```

Uses the `penn_state_rivendale_seg` dataset. Outputs pixel-level instance masks in addition to bounding boxes. Edit the dataset path and model config at the top of the script.

---

## Inference

### Single image with ground-truth overlay

```bash
python infer.py
```

Edit `img_path` and `weights` inside the script. Displays predicted boxes (color-coded by class) overlaid with ground-truth boxes if a matching YOLO `.txt` label file is found alongside the image.

### Batch inference

```bash
python infer_batch.py
```

Runs inference on a directory of images. Edit the `SOURCE` directory and `weights` path inside the script. Outputs are saved to `runs/detect/predict*/`.

### Streamlit web UI

```bash
streamlit run infer_streamlit.py
```

Launches a browser-based interface supporting image upload, webcam snapshot, and live OpenCV stream modes.

---

## Validation

### Standard validation

```bash
python val.py
```

Runs `model.val()` and saves mAP, precision, recall, PR curves, and confusion matrix plots to `runs/detect/<name>/`.

### 25-channel model validation

```bash
python y8_mosaic_val.py
```

Same as `val.py` but uses the custom mosaic validator that handles 25-channel input and optional band subset selection. Edit `MODEL` and `DATASET` at the top.

### Configurable IoU threshold validation

```bash
python custom_detection_validator.py
```

Evaluates mAP at multiple custom IoU thresholds (e.g., mAP60, mAP70, mAP80) and generates TP/FP/FN overlays for each threshold. Useful for understanding detection quality beyond the standard mAP50-95.

---

## Band Importance Analysis

After training a 25-channel model with `y8_mosaic_trainer.py`, extract per-band importance scores from the first convolutional layer weights:

```bash
python model_analyze.py \
    --weights runs/train/<run_name>/weights/best.pt \
    --plot
```

Outputs written to `band_analysis/`:
- `first_conv_band_importance.csv` — bands ranked by L1 weight norm
- `first_conv_band_importance.json` — same data as JSON
- `first_conv_per_filter_preference.npy` — `[C_out × 25]` per-filter band preference matrix
- `ranked_band_importance.png`, `top_10_bands.png` — bar chart visualizations

**Result:** NIR bands at **836 nm, 824 nm, and 926 nm** show the highest importance, consistent with plant chlorophyll and water absorption features relevant to disease stress.

---

## ONNX Export

See [ONNX_export.md](ONNX_export.md) for full instructions.

Quick reference — export with raw model outputs (no built-in NMS, for flexible C++ post-processing):

```bash
yolo export model=/path/to/best.pt \
    format=onnx \
    imgsz=1088,1440 \
    opset=17 \
    simplify=True \
    dynamic=False \
    nms=False
```

---

## Helper Scripts

| Script | Description |
|---|---|
| `helpers/batch_demosaic.py` | Converts raw Ximea mosaic (.pgm) dataset to 25-channel demosaiced images (.png) and adjusts YOLO labels for the coordinate change |
| `helpers/make_val.py` | Splits a flat dataset into train/val sets (80/20 default) |
| `helpers/cleanup_labels.py` | Removes orphan images (no label) and orphan labels (no image); rebuilds `train.txt` |
| `helpers/remap.py` | Remaps class IDs (e.g., merge multiple canker subtypes into a single class); updates `data.yaml` |
| `helpers/clahe.py` | Applies CLAHE + gamma adjustment + saturation boost to RGB images |
| `helpers/histogram_matching.py` | Transfers color/intensity distribution from a source image to a target (LAB-space domain adaptation) |
| `helpers/get_gt_labels.py` | Visualizes ground-truth YOLO annotations overlaid on images |
| `helpers/mosaic_with_labels.py` | Visualizes raw Bayer mosaic data with annotation overlays |
| `helpers/lab_color.py` | Decomposes images into LAB color channels for inspection |
| `helpers/image_matching.py` | Feature-based image matching (for dataset alignment) |
