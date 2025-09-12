#!/usr/bin/env python3
import argparse
from ultralytics import YOLO
import cv2
import os
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

        # Load image (keep BGR for correct OpenCV color usage) and size
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        H, W = image_bgr.shape[:2]

        # Helper: try to locate a YOLO .txt label file for this image
        def find_label_file(img_path):
            stem, _ = os.path.splitext(img_path)
            candidates = []
            # Common: images/... -> labels/....txt
            candidates.append(stem + ".txt")
            # replace 'images' or 'image' segments with 'labels'
            candidates.append(img_path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep).rsplit('.', 1)[0] + ".txt")
            candidates.append(img_path.replace(os.sep + "image" + os.sep, os.sep + "labels" + os.sep).rsplit('.', 1)[0] + ".txt")
            # sibling 'labels' directory next to the image folder: ../labels/<name>.txt
            candidates.append(os.path.join(os.path.dirname(os.path.dirname(img_path)), "labels", os.path.basename(stem) + ".txt"))
            # labels directory inside the same parent as the image (e.g., dataset/labels/<name>.txt)
            candidates.append(os.path.join(os.path.dirname(img_path).rsplit(os.sep, 1)[0], "labels", os.path.basename(stem) + ".txt"))
            # also labels in a 'labels' subfolder alongside the image (less common)
            candidates.append(os.path.join(os.path.dirname(img_path), "labels", os.path.basename(stem) + ".txt"))
            for c in candidates:
                if os.path.exists(c):
                    return c
            return None

        # Load ground truth boxes if label file exists (YOLO format: cls cx cy w h normalized)
        gt_boxes = []  # list of (x1,y1,x2,y2,cls_id)
        label_file = find_label_file(img_path)
        if label_file is not None:
            print(f"[infer.py] Using label file: {label_file}")
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(float(parts[0]))
                    cx = float(parts[1]) * W
                    cy = float(parts[2]) * H
                    bw = float(parts[3]) * W
                    bh = float(parts[4]) * H
                    x1 = int(max(0, cx - bw / 2))
                    y1 = int(max(0, cy - bh / 2))
                    x2 = int(min(W - 1, cx + bw / 2))
                    y2 = int(min(H - 1, cy + bh / 2))
                    gt_boxes.append((x1, y1, x2, y2, cls_id))
            print(f"[infer.py] Parsed {len(gt_boxes)} GT boxes: {gt_boxes}")
        else:
            # Helpful debug info when GT labels can't be found
            print(f"[infer.py] No label file found for image: {img_path}\nSearched common locations. If your labels live in a different folder, update find_label_file().")

        # Define a fixed 4-color palette (BGR tuples). Only these 4 colors will be used.
        palette = [
            (0, 0, 255),    # red
            (0, 255, 0),    # green
            (255, 0, 0),    # blue
            (0, 255, 255),  # yellow
        ]

        # Map each class to two colors (pred, gt) using only the 4-color palette.
        # pred_color = palette[(cls_id * 2) % 4]
        # gt_color   = palette[(cls_id * 2 + 1) % 4]

        # Draw prediction boxes (BGR) on image_bgr
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            color = palette[(cls_id * 2) % 4]
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color=color, thickness=3)

        # Draw GT boxes (if any) with the paired color
        for (x1, y1, x2, y2, cls_id) in gt_boxes:
            color = palette[(cls_id * 2 + 1) % 4]
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color=color, thickness=3)
            # draw a small caption for the GT class to make it clear
            cls_name = class_names.get(cls_id, str(cls_id)) if isinstance(class_names, dict) else str(cls_id)
            cv2.putText(image_bgr, f"GT:{cls_name}", (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=1, lineType=cv2.LINE_AA)

        # Build legend: for each seen class, show Pred:Class and GT:Class with their colors
        pred_cls_set = set(int(box.cls[0]) for box in boxes)
        gt_cls_set = set([c for (_, _, _, _, c) in gt_boxes])
        seen_cls = sorted(pred_cls_set | gt_cls_set)

        legend_elements = []
        for cls_id in seen_cls:
            # robust class name lookup
            if isinstance(class_names, dict):
                name = class_names.get(cls_id, str(cls_id))
            else:
                try:
                    name = class_names[cls_id]
                except Exception:
                    name = str(cls_id)

            pred_col = np.array(palette[(cls_id * 2) % 4][::-1]) / 255.0  # BGR->RGB for matplotlib
            gt_col = np.array(palette[(cls_id * 2 + 1) % 4][::-1]) / 255.0
            legend_elements.append(Patch(facecolor=pred_col, edgecolor='black', label=f"Pred: {name}"))
            legend_elements.append(Patch(facecolor=gt_col,   edgecolor='black', label=f"GT:   {name}"))

        # Convert to RGB for matplotlib display
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Plot image + legend
        plt.figure(figsize=(12, 10))
        plt.imshow(image_rgb)
        plt.title(f"Detected {len(boxes)} objects")
        plt.axis('off')
        if legend_elements:
            plt.legend(handles=legend_elements, loc='upper right', title="Legend")
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
    p.add_argument("--conf", type=float, default=0.1, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.1, help="NMS IoU threshold")
    p.add_argument("--use_key", type=bool, default=True, help="Use key for inference")
    args = p.parse_args()

    infer_image(
        img_path=args.img,
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        use_key=args.use_key
    )