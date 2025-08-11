#!/usr/bin/env python3
import torch
from ultralytics import YOLO
import ultralytics.nn
from torch.nn.modules.container import Sequential

def train_yolov11(
    model_cfg: str = "yolo11l.pt",
    data_cfg:  str = "rivendale_uniques_reduced_ximea_demosaic/data.yaml",
    project:   str = "runs/train",
    name:      str = "yolov11_large_uniques_reduced_ximea_demosaic",
    continue_training: bool = False
):
    if continue_training:
        model = YOLO(f"runs/train/{name}/weights/last.pt")
    else:
        model = YOLO(model_cfg)

    # Train the model
    model.train(
        data        = data_cfg,
        epochs      = 100,
        imgsz       = (1088, 1440),
        batch       = 3,
        patience    = 30,
        lr0         = 0.01,
        lrf         = 0.01,
        dropout     = 0.1,
        augment     = True,
        # hsv_h       = 0.05,
        # hsv_s       = 0.5,
        # hsv_v       = 0.5,
        mosaic      = 1.0,
        # mixup       = 0.2,
        project     = project,
        name        = name,
        exist_ok    = True,
        resume      = True if continue_training else False
    )

if __name__ == "__main__":
    train_yolov11(continue_training=False)
