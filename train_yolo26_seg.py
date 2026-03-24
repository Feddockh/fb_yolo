#!/usr/bin/env python3
import os
from ultralytics import YOLO

pwd = os.path.dirname(os.path.abspath(__file__))

# Pick a YOLO26 segmentation checkpoint (n/s/m/l/x depending on your GPU + needs).
# Official YOLO26 segmentation models use the "-seg" suffix.
MODEL = "yolo26m-seg.pt"   # try: yolo26n-seg.pt, yolo26s-seg.pt, yolo26m-seg.pt, ...
# MODEL = "/home/hayden/cmu/kantor_lab/fb_models/fb_yolo/runs/train/yolo26_large_seg_rivendale_v6/weights/best.pt"
# MODEL = "/home/hayden/cmu/kantor_lab/fb_models/fb_yolo/runs/train/yolo26_large_seg_pybullet_fb_images_real_finetune/weights/last.pt"

# DATASET = "lab_fb_images_v2_seg"
# DATASET = "rivendale_v6_seg"
# DATASET = "pybullet_fb_images_v2_seg"
# DATASET = "gazebo_fb_images_seg"
# DATASET = "penn_state_seg"
DATASET = "penn_state_rivendale_seg"
OUTPUT_DIR = "runs/train"

def train_yolo26_seg(
    model_cfg: str = os.path.join(pwd, "models", MODEL),
    data_cfg:  str = os.path.join(pwd, "datasets", DATASET, "data.yaml"),
    project:   str = os.path.join(pwd, OUTPUT_DIR),
    name:      str = "yolo26_large_penn_state_rivendale_seg_v3",
):
    # Load pretrained YOLO26 segmentation weights
    # model = YOLO(model_cfg)
    model = YOLO(MODEL)

    model.train(
        data         = data_cfg,
        # task         = "segment",     # make intent explicit (usually inferred from -seg)
        epochs       = 200,
        imgsz        = (1088, 1440),
        batch        = 3,
        patience     = 100,
        lr0          = 0.01,
        lrf          = 0.01,
        dropout      = 0.0,
        augment      = True,
        # resume       = True,

        # Augmentations
        hsv_h        = 0.015,
        hsv_s        = 0.7,
        hsv_v        = 0.4,
        degrees      = 0.0,
        translate    = 0.1, # tried 0.05, was worse
        scale        = 0.5, # tried 0.9, was worse
        shear        = 0.0,
        perspective  = 0.0002,
        flipud       = 0.0,
        fliplr       = 0.5,
        mosaic       = 1.0, # tried 0.5, was worse
        close_mosaic = 30,
        mixup        = 0.0,
        copy_paste   = 0.0,

        # Regularization / schedule
        weight_decay  = 0.0005,
        warmup_epochs = 3.0,
        cos_lr        = False,

        # YOLO26 is end-to-end and NMS-free by default.
        # Keep end2end=True unless you have a specific reason to validate with NMS.
        end2end      = True,

        project      = project,
        name         = name,
        exist_ok     = True,
        plots        = True,
    )

if __name__ == "__main__":
    # Match your behavior: cd into dataset directory
    os.chdir(os.path.join(pwd, "datasets", DATASET))
    print("CWD:", os.getcwd())

    train_yolo26_seg()
