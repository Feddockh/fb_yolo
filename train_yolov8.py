#!/usr/bin/env python3
import os
import torch
from ultralytics import YOLO
import ultralytics.nn
from torch.nn.modules.container import Sequential

pwd = os.path.dirname(os.path.abspath(__file__))

MODEL = "yolov8l.pt"
DATASET = "lab_fb_images"
OUTPUT_DIR = "runs/train"

def train_yolov8(
    model_cfg: str = os.path.join(pwd, "models", MODEL),
    data_cfg:  str = os.path.join(pwd, "datasets", DATASET, "data.yaml"),
    project:   str = os.path.join(pwd, OUTPUT_DIR),
    name:      str = "yolov8_large_lab_fb_images_v2",
):

    # model = YOLO(model=model_cfg)
    model = YOLO(os.path.join(pwd, "runs/train/yolov8_large_rivendale_v6_k_fold/yolov8_large_rivendale_v6_k_fold1/weights/best.pt"))

    model.train(
        data        = data_cfg,
        epochs      = 200,
        imgsz       = (1088, 1440),
        # imgsz       = (216, 409), # For Ximea camera (demosaic)
        # imgsz       = (1088, 2048), # For Ximea camera
        batch       = -1, # Auto batch size based on GPU memory
        # patience    = 200,
        patience    = 100, # Default
        lr0         = 0.01, # Default
        lrf         = 0.01, # Default
        # dropout     = 0.2,  # Increase dropout to prevent overfitting
        dropout     = 0.0, # Default
        augment     = True, # Default
        # hsv_h       = 0.0,
        # hsv_s       = 0.0,
        # hsv_v       = 0.0,
        hsv_h       = 0.015,  # Default hue augmentation
        hsv_s       = 0.7,    # Default saturation augmentation
        hsv_v       = 0.4,    # Default value augmentation
        # degrees     = 15.0,
        degrees     = 0.0,    # Default random rotation augmentation
        translate   = 0.1,    # Default random translation augmentation
        scale       = 0.5,    # Default random scaling augmentation
        # shear       = 2.0,
        shear       = 0.0,    # Default random shear augmentation
        perspective = 0.0002, # Default random Perspective transformation
        flipud      = 0.0,    # Default vertical flip probability
        fliplr      = 0.5,    # Default horizontal flip probability
        mosaic      = 1.0,    # Default mosaic augmentation
        mixup       = 0.0,    # Default mixup augmentation
        copy_paste  = 0.0,    # Default copy-paste augmentation
        # Regularization parameters
        weight_decay = 0.0005, # Default L2 regularization
        warmup_epochs = 3.0,   # Default gradual learning rate warmup
        # cos_lr      = True,
        cos_lr      = False,   # Default step learning rate scheduler
        project     = project,
        name        = name,
        exist_ok    = True,
        # resume      = True,
        plots       = True,

        iou         = 0.3,
        # iou         = 0.7,    # Default NMS IoU threshold
        conf        = 0.1,
        # conf        = 0.001,   # Default NMS confidence threshold
        nms         = True
    )

if __name__ == "__main__":
    # Change to the dataset directory
    os.chdir(os.path.join(pwd, "datasets", DATASET))
    print(os.getcwd())

    train_yolov8()
