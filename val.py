#!/usr/bin/env python3
from ultralytics import YOLO

def test_yolov8(
    model_cfg: str = "runs/train/yolov8_fireblight_large_remapped/weights/best.pt",
    data_cfg:  str = "batch_1_remapped/data.yaml",
    imgsz:     tuple = (1088, 1440)
):
    model = YOLO(model_cfg)

    # Run evaluation on the test set defined in data.yaml
    metrics = model.val(
        data = data_cfg,
        split = "test",
        imgsz = imgsz,
    )

    # class_index = metrics.names.index("Shepherd's Crook")  # or your target class

    # map50_for_target = metrics.box.map50_per_class[class_index]
    # print(f"mAP@0.5 for 'Shepherd's Crook': {map50_for_target:.3f}")

    # Optional: Print or save results
    print(metrics)

if __name__ == "__main__":
    test_yolov8()