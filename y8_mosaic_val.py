#!/usr/bin/env python3
import os
import torch
from ultralytics import YOLO
import ultralytics.nn
from torch.nn.modules.container import Sequential
import y8_mosaic_trainer
from y8_mosaic_trainer import YOLOMosaic, MosaicModel, DetectionModelMosaic, Mosaic25Validator, BANDWIDTHS

pwd = os.path.dirname(os.path.abspath(__file__))

MODEL = "yolov8l_25_band_input_experiment"
DATASET = "rivendale_v6_ximea_bandgate"
OUTPUT_DIR = "runs/detect"
# BANDS = [806, 757, 793]
BANDS = BANDWIDTHS.ravel().tolist()

def validate_yolov8(
    model_cfg: str = os.path.join(pwd, "runs", "train", f"{MODEL}", "weights", "best.pt"),
    data_cfg:  str = os.path.join(pwd, "datasets", DATASET, "data.yaml"),
    project:   str = os.path.join(pwd, OUTPUT_DIR),
    name:      str = f"{MODEL}",
    bands:    list = BANDS,
):
    y8_mosaic_trainer.USING_BANDS = bands
    print(f"Using bands: {bands}")

    model = YOLOMosaic(model_cfg)

    # Output the model architecture
    print(model)

    # Run validation
    model.val(
        data        = data_cfg,
        imgsz       = (216, 409),
        batch       = 1,
        workers     = 1,
        conf        = 0.1,   # NMS confidence threshold
        iou         = 0.3,   # NMS IoU threshold
        max_det     = 300,    # Max detections per image
        agnostic_nms= False,  # Per-class NMS
        plots       = True,   # Save PR/confusion matrix plots
        save_json   = False,  # Save COCO-format JSON (optional)
        verbose     = True,
        name        = name,
        project     = project,
    )

if __name__ == "__main__":
    # os.chdir(os.path.join(pwd, "datasets", DATASET))
    # print(os.getcwd())

    validate_yolov8()
