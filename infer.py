#!/usr/bin/env python3
import argparse
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


def infer_image(
    img_path: str,
    weights: str = "runs/train/yolov8_large_uniques/weights/best.pt",
    conf:    float = 0.25,
    iou:     float = 0.45,
    use_key: bool = True
):
    if use_key:
        model = YOLO(weights)
        results = model(img_path, conf=conf, iou=iou)[0]
        boxes = results.boxes
        class_names = model.names

        # Load and convert image
        image_bgr = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Generate unique colors per class
        np.random.seed(42)  # for consistent colors
        color_map = {
            cls_id: tuple(np.random.randint(0, 255, size=3).tolist())
            for cls_id in set(int(box.cls[0]) for box in boxes)
        }

        # Draw bounding boxes without labels
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            color = color_map[cls_id]
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color=color, thickness=2)

        # Create legend handles
        legend_elements = [
            Patch(facecolor=np.array(color_map[cls_id]) / 255.0,
                edgecolor='black',
                label=class_names[cls_id])
            for cls_id in sorted(color_map.keys())
        ]

        # Plot image + legend
        plt.figure(figsize=(12, 10))
        plt.imshow(image_rgb)
        plt.title(f"Detected {len(boxes)} objects")
        plt.axis('off')
        plt.legend(handles=legend_elements, loc='upper right', title="Class Legend")
        plt.tight_layout()
        plt.show()

    else:

        # 1. Load model
        model = YOLO(weights)

        # 2. Run inference
        results = model(img_path, conf=conf, iou=iou)[0]

        # 3. Get annotated image (NumPy array BGR)
        annotated = results.plot()  # H×W×3 numpy array

        # 4. Show with matplotlib
        # convert BGR to RGB for correct color display in matplotlib
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        plt.imshow(annotated_rgb)
        plt.title(f"Detected {len(results.boxes)} objects")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img", type=str, required=True, help="Path to input image")
    p.add_argument("--weights", type=str, default="runs/train/yolov8_fireblight_large_remapped/weights/best.pt",
                   help="Path to model weights (best.pt)")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--use_key", type=bool, default=True, help="Use key for inference")
    args = p.parse_args()

    infer_image(
        img_path=args.img,
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        use_key=args.use_key
    )