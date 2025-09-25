#!/usr/bin/env python3
import torch
from ultralytics import YOLO
import ultralytics.nn
from torch.nn.modules.container import Sequential

def train_rtdetr(
    model_cfg: str = "rtdetr-l.pt",
    data_cfg:  str = "rivendale_uniques/data.yaml",
    project:   str = "runs/train",
    name:      str = "rtdetr_nano",
):
    model = YOLO(model_cfg)
    # model = YOLO("runs/train/yolov8_fireblight_large_uniques_remapped/weights/last.pt")

    # Train the model
    model.train(
        data        = data_cfg,
        epochs      = 200,
        imgsz       = (1088, 1440),
        batch       = 2,
        patience    = 20,  # Reduced from 30 - stop earlier if no improvement
        lr0         = 0.001,
        lrf         = 0.001,
        dropout     = 0.2,  # Increased dropout to reduce overfitting
        augment     = True,
        # Enhanced data augmentation to prevent overfitting
        hsv_h       = 0.015,  # Hue augmentation
        hsv_s       = 0.7,    # Saturation augmentation  
        hsv_v       = 0.4,    # Value augmentation
        degrees     = 15.0,   # Random rotation
        translate   = 0.1,    # Random translation
        scale       = 0.5,    # Random scaling
        shear       = 2.0,    # Random shear
        perspective = 0.0002, # Perspective transformation
        flipud      = 0.0,    # Vertical flip probability
        fliplr      = 0.5,    # Horizontal flip probability
        mosaic      = 1.0,    # Keep mosaic augmentation
        mixup       = 0.15,   # Add mixup augmentation
        copy_paste  = 0.3,    # Add copy-paste augmentation
        # Regularization parameters
        weight_decay = 0.0005, # L2 regularization
        warmup_epochs = 3.0,   # Gradual learning rate warmup
        cos_lr      = True,    # Cosine learning rate scheduler
        project     = project,
        name        = name,
        exist_ok    = True,
        # resume      = True,
    )

if __name__ == "__main__":
    train_rtdetr()
