#!/usr/bin/env python3
import argparse
import os
import glob
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


def draw_gt_boxes(image, label_path, class_names, color_map):
    """Draws ground truth bounding boxes on an image."""
    if not os.path.exists(label_path):
        return image, []

    h, w, _ = image.shape
    gt_classes = set()
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            gt_classes.add(class_id)
            x_center, y_center, width, height = map(float, parts[1:])
            
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)
            
            color = color_map.get(class_id, (255, 255, 255)) # Default to white if color not in map
            cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=2)
            
    return image, list(gt_classes)


def infer_and_compare_image(
    img_path: str,
    label_path: str,
    out_path: str,
    model: YOLO,
    conf: float = 0.25,
    iou: float = 0.45
):
    # --- 1. Model Inference ---
    results = model(img_path, conf=conf, iou=iou)[0]
    boxes = results.boxes
    class_names = model.names

    # --- 2. Prepare Images ---
    image_bgr = cv2.imread(img_path)
    pred_image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gt_image_rgb = pred_image_rgb.copy()

    # --- 3. Define Colors ---
    # Use fixed colors for consistency across images
    fixed_colors = {
        0: (0, 255, 0),    # green
        1: (0, 0, 255),    # blue
        2: (255, 255, 0),  # yellow
        3: (255, 0, 0),    # red
        4: (0, 255, 255),  # cyan
        5: (255, 0, 255),  # magenta
        6: (128, 0, 128)   # purple
    }
    
    # --- 4. Draw Prediction Boxes ---
    pred_classes = {int(box.cls[0]) for box in boxes}
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        color = fixed_colors.get(cls_id, (255, 255, 255))
        cv2.rectangle(pred_image_rgb, (x1, y1), (x2, y2), color=color, thickness=2)

    # --- 5. Draw Ground Truth Boxes ---
    gt_image_rgb, gt_classes = draw_gt_boxes(gt_image_rgb, label_path, class_names, fixed_colors)

    # --- 6. Create Legend ---
    all_classes = sorted(list(pred_classes.union(gt_classes)))
    legend_elements = [
        Patch(facecolor=np.array(fixed_colors.get(cls_id, (255,255,255))) / 255.0,
              edgecolor='black',
              label=class_names.get(cls_id, f"Unknown: {cls_id}"))
        for cls_id in all_classes
    ]

    # --- 7. Plot and Save ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    
    ax1.imshow(pred_image_rgb)
    ax1.set_title(f"Predictions ({len(boxes)} objects)")
    ax1.axis('off')
    
    ax2.imshow(gt_image_rgb)
    ax2.set_title("Ground Truth")
    ax2.axis('off')
    
    fig.legend(handles=legend_elements, loc='upper right', title="Class Legend")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run inference on a directory of images and compare with ground truth.")
    p.add_argument("--img_dir", type=str, required=True, help="Path to input image directory (e.g., 'data/images/val')")
    p.add_argument("--label_dir", type=str, required=True, help="Path to ground truth label directory (e.g., 'data/labels/val')")
    p.add_argument("--out_dir", type=str, required=True, help="Path to output directory for comparison images")
    p.add_argument("--weights", type=str, default="runs/train/yolov8_fireblight_large_remapped/weights/best.pt",
                   help="Path to model weights (best.pt)")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    image_paths = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        image_paths.extend(glob.glob(os.path.join(args.img_dir, ext)))

    model = YOLO(args.weights)

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(args.label_dir, label_name)
        
        out_path = os.path.join(args.out_dir, img_name)
        
        print(f"Processing {img_path}...")
        infer_and_compare_image(
            img_path=img_path,
            label_path=label_path,
            out_path=out_path,
            model=model,
            conf=args.conf,
            iou=args.iou
        )
    print(f"Processing complete. Output saved to {args.out_dir}")
