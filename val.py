#!/usr/bin/env python3
import os
import torch
from ultralytics import YOLO
import ultralytics.nn
from torch.nn.modules.container import Sequential

pwd = os.path.dirname(os.path.abspath(__file__))

MODEL = "yolo26_large_penn_state_rivendale_seg"
DATASET = "penn_state_rivendale_seg"
OUTPUT_DIR = "runs/detect"

def validate_yolov8(
    model_cfg: str = os.path.join(pwd, "runs", "train", f"{MODEL}", "weights", "best.pt"),
    data_cfg:  str = os.path.join(pwd, "datasets", DATASET, "data.yaml"),
    project:   str = os.path.join(pwd, OUTPUT_DIR),
    name:      str = f"{MODEL}"
):
    model = YOLO(model_cfg)

    # Run validation
    model.val(
        data        = data_cfg,
        imgsz       = (1088, 1440),
        batch       = 1,
        workers     = 1,
        conf        = 0.1,   # NMS confidence threshold
        iou         = 0.1,   # NMS IoU threshold
        max_det     = 300,    # Max detections per image
        agnostic_nms= False,  # Per-class NMS
        plots       = True,   # Save PR/confusion matrix plots
        save_json   = False,  # Save COCO-format JSON (optional)
        verbose     = True,
        name        = name,
        project     = project,
    )

if __name__ == "__main__":
    os.chdir(os.path.join(pwd, "datasets", DATASET))
    print(os.getcwd())

    validate_yolov8()
