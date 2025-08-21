#!/usr/bin/env python3
import argparse
import os
import glob
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


def draw_boxes(image, boxes, class_names, color_map, is_gt=False):
    """Draws bounding boxes (GT or prediction) on an image."""
    h, w, _ = image.shape
    
    # Define brightness factors for GT and predictions
    gt_factor = 0.5  # Darker
    pred_factor = 1.8 # Lighter

    for box_data in boxes:
        if is_gt:
            class_id, x_center, y_center, width, height = box_data
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)
            label = "" # No label for GT boxes
        else: # It's a prediction
            x1, y1, x2, y2 = map(int, box_data.xyxy[0])
            class_id = int(box_data.cls[0])
            confidence = float(box_data.conf[0])
            label = f"{confidence:.2f}"

        color = np.array(color_map.get(class_id, (255, 255, 255)))
        
        # Adjust color brightness
        if is_gt:
            # Make color darker for GT
            draw_color = (color * gt_factor).astype(int).tolist()
        else:
            # Make color lighter for predictions
            draw_color = np.clip(color * pred_factor, 0, 255).astype(int).tolist()

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color=draw_color, thickness=2)
        
        # Prepare and draw the label if it exists
        if label:
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x1, y1 - text_h - 5), (x1 + text_w, y1), draw_color, -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    return image


def infer_and_visualize_combined(
    img_path: str,
    label_path: str,
    out_path: str,
    model: YOLO,
    conf: float = 0.25,
    iou: float = 0.45
):
    # --- 1. Model Inference ---
    results = model(img_path, conf=conf, iou=iou, nms=True, verbose=False)[0]
    pred_boxes = results.boxes
    class_names = model.names

    # --- 2. Prepare Image ---
    image_bgr = cv2.imread(img_path)
    combined_image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # --- 3. Define Base Colors ---
    color_map = {
        0: (0, 255, 0),    # green for Shepherd's Crook
        1: (0, 0, 255),    # blue for Canker
        # Add more colors if you have more classes
    }

    # --- 4. Read and Draw Ground Truth Boxes ---
    gt_boxes_data = []
    gt_classes = set()
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                gt_classes.add(class_id)
                coords = list(map(float, parts[1:]))
                gt_boxes_data.append([class_id] + coords)
        combined_image_rgb = draw_boxes(combined_image_rgb, gt_boxes_data, class_names, color_map, is_gt=True)

    # --- 5. Draw Prediction Boxes ---
    pred_classes = {int(box.cls[0]) for box in pred_boxes}
    combined_image_rgb = draw_boxes(combined_image_rgb, pred_boxes, class_names, color_map, is_gt=False)

    # --- 6. Create Legend ---
    all_classes = sorted(list(pred_classes.union(gt_classes)))
    legend_elements = []
    for cls_id in all_classes:
        class_name = class_names.get(cls_id, f"Unknown: {cls_id}")
        color = np.array(color_map.get(cls_id, (255,255,255))) / 255.0
        
        # GT legend entry (darker)
        gt_color = tuple(np.clip(color * 0.5, 0, 1))
        legend_elements.append(Patch(facecolor=gt_color, edgecolor='black', label=f"{class_name} (GT)"))
        
        # Prediction legend entry (lighter)
        pred_color = tuple(np.clip(color * 1.8, 0, 1))
        legend_elements.append(Patch(facecolor=pred_color, edgecolor='black', label=f"{class_name} (Pred)"))


    # --- 7. Plot and Save ---
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    
    ax.imshow(combined_image_rgb)
    ax.set_title(f"Ground Truth (Dark) vs. Predictions (Light) - {os.path.basename(img_path)}")
    ax.axis('off')
    
    fig.legend(handles=legend_elements, loc='upper right', title="Legend", fontsize='large')
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for legend
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run inference and create a combined visualization of predictions and ground truth.")
    p.add_argument("--img_dir", type=str, required=True, help="Path to input image directory (e.g., 'data/images/val')")
    p.add_argument("--label_dir", type=str, required=True, help="Path to ground truth label directory (e.g., 'data/labels/val')")
    p.add_argument("--out_dir", type=str, required=True, help="Path to output directory for combined images")
    p.add_argument("--weights", type=str, default="runs/train/yolov8_fireblight_large_remapped/weights/best.pt",
                   help="Path to model weights (best.pt)")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    image_paths = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        image_paths.extend(glob.glob(os.path.join(args.img_dir, ext)))

    model = YOLO(args.weights)
    total_images = len(image_paths)

    for i, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(args.label_dir, label_name)
        
        out_path = os.path.join(args.out_dir, img_name)
        
        print(f"Processing image {i+1}/{total_images}: {img_path}...")
        infer_and_visualize_combined(
            img_path=img_path,
            label_path=label_path,
            out_path=out_path,
            model=model,
            conf=args.conf,
            iou=args.iou
        )
    print(f"Processing complete. Output saved to {args.out_dir}")
