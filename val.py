#!/usr/bin/env python3
import torch
from ultralytics import YOLO
import ultralytics.nn
from torch.nn.modules.container import Sequential

def validate_yolov8(
    model_cfg: str = "runs/train/yolov8_large_rivendale_v4_remapped/weights/best.pt",
    data_cfg:  str = "rivendale_v4_remapped/data.yaml",
    name:      str = "yolov8_large_rivendale_v4_remapped"
):
    model = YOLO(model_cfg)

    # Run validation
    model.val(
        data        = data_cfg,
        imgsz       = (1088, 1440),
        batch       = 2,
        workers     = 8,
        conf        = 0.25,   # NMS confidence threshold
        iou         = 0.20,   # NMS IoU threshold
        max_det     = 300,    # Max detections per image
        agnostic_nms= False,  # Per-class NMS
        plots       = True,   # Save PR/confusion matrix plots
        save_json   = False,  # Save COCO-format JSON (optional)
        verbose     = True,
        name        = name,
    )

if __name__ == "__main__":
    validate_yolov8()
