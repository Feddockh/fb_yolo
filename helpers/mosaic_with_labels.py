#!/usr/bin/env python3
import argparse
import random
import math
import os
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image
from ultralytics import YOLO

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def yolo_to_xyxy(xc: float, yc: float, w: float, h: float, W: int, H: int) -> Tuple[float, float, float, float]:
    """Convert YOLO-normalized (xc, yc, w, h) to pixel (x1, y1, x2, y2)."""
    x1 = (xc - w / 2.0) * W
    y1 = (yc - h / 2.0) * H
    x2 = (xc + w / 2.0) * W
    y2 = (yc + h / 2.0) * H
    return x1, y1, x2, y2


def infer_label_path(img_path: str) -> str:
    """
    Build a label path by swapping 'images' -> 'labels' and '.png' -> '.txt'.
    Includes a couple of safe fallbacks if that file doesn't exist.
    """
    label_path = img_path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
    root, _ext = os.path.splitext(label_path)
    label_path = root + ".txt"

    if not os.path.exists(label_path):
        # Fallback 1: 'image' (singular) → 'labels'
        alt = img_path.replace(os.sep + "image" + os.sep, os.sep + "labels" + os.sep)
        alt = os.path.splitext(alt)[0] + ".txt"
        if os.path.exists(alt):
            return alt

        # Fallback 2: same dir as image, just swap extension
        alt2 = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(alt2):
            return alt2

    return label_path


def render_boxes_with_matplotlib(
    img_pil: Image.Image,
    gt_boxes: List[Tuple[float, float, float, float, int]],
    pred_boxes: List[Tuple[float, float, float, float, int]],
    gt_colors: List[str],
    pred_colors: List[str],
    line_width: int,
) -> Image.Image:
    """
    Draw rectangles using Matplotlib and return a PIL Image with the annotations rendered.
    gt_boxes/pred_boxes: list of (x1, y1, x2, y2, class_id)
    """
    W, H = img_pil.size
    dpi = 200
    fig_w = W / dpi
    fig_h = H / dpi

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # full-bleed image, no margins
    ax.imshow(img_pil)
    ax.set_axis_off()

    # Draw GT (solid)
    for (x1, y1, x2, y2, cid) in gt_boxes:
        color = gt_colors[cid % len(gt_colors)]
        ax.add_patch(
            patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=line_width,
                edgecolor=color,
                facecolor='none',
            )
        )

    # Draw predictions (dashed)
    for (x1, y1, x2, y2, cid) in pred_boxes:
        color = pred_colors[cid % len(pred_colors)]
        ax.add_patch(
            patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=line_width,
                edgecolor=color,
                facecolor='none',
                linestyle='--',
            )
        )

    # Render to RGBA buffer (backend-safe)
    fig.canvas.draw()
    buf, (w, h) = fig.canvas.print_to_buffer()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)  # RGBA
    plt.close(fig)

    # Drop alpha channel -> RGB
    rgb = arr[:, :, :3]
    return Image.fromarray(rgb)



def annotate_single_image(
    img_path: str,
    gt_colors: List[str],
    pred_colors: List[str],
    class_names: List[str],
    model: Optional[YOLO],
    iou: float,
    conf: float,
    target_size: Optional[Tuple[int, int]],
    line_width: int,
) -> Image.Image:
    """
    Load one image, gather GT + predicted boxes, draw them with Matplotlib, return annotated PIL image.
    """
    img = Image.open(img_path).convert("RGB")
    if target_size is not None and img.size != target_size:
        img = img.resize(target_size)
    W, H = img.size

    # ----- Gather GT boxes -----
    gt_boxes: List[Tuple[float, float, float, float, int]] = []
    label_path = infer_label_path(img_path)
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])
                x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, W, H)
                gt_boxes.append((x1, y1, x2, y2, class_id))

    # ----- Gather predicted boxes -----
    pred_boxes: List[Tuple[float, float, float, float, int]] = []
    if model is not None:
        res = model(img, iou=iou, conf=conf)[0]
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cls_id = int(b.cls[0])
            pred_boxes.append((x1, y1, x2, y2, cls_id))

    # Render with Matplotlib
    annotated = render_boxes_with_matplotlib(
        img_pil=img,
        gt_boxes=gt_boxes,
        pred_boxes=pred_boxes,
        gt_colors=gt_colors,
        pred_colors=pred_colors,
        line_width=line_width,
    )
    return annotated


def create_mosaic(
    image_files: List[str],
    n: int,
    output_file: str,
    seed: Optional[int] = None,
    weights_path: Optional[str] = None,
    iou: float = 0.3,
    conf: float = 0.1,
    gt_colors: Optional[List[str]] = None,
    pred_colors: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    line_width: int = 2,
):
    """
    Annotates each image individually (GT + predictions with Matplotlib), then assembles them into a mosaic.
    Legend/key is added with Matplotlib (clean, outside the image).
    """
    if seed is not None:
        random.seed(seed)

    if n > len(image_files):
        print(f"Warning: Requested {n} images, but only {len(image_files)} are available. Using all available images.")
        n = len(image_files)

    if n == 0:
        print("No images to create a mosaic.")
        return

    random_images = random.sample(image_files, n)

    # Load first image to define grid tile size
    try:
        with Image.open(random_images[0]) as img0:
            tile_w, tile_h = img0.size
    except Exception as e:
        print(f"Could not open the first image to get dimensions: {e}")
        return

    # Grid layout
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    mosaic_w = cols * tile_w
    mosaic_h = rows * tile_h

    # Prepare model once (if provided)
    model = None
    if weights_path:
        if os.path.exists(weights_path):
            print("Loading model weights...")
            model = YOLO(weights_path)
        else:
            print(f"Error: Weights file not found at {weights_path}")

    # Defaults
    if class_names is None:
        class_names = ["Shepherd's Crook", "Canker"]
    if gt_colors is None:
        gt_colors = ["green", "red"]
    if pred_colors is None:
        pred_colors = ["blue", "orange"]

    # Build the mosaic from individually annotated tiles
    mosaic = Image.new("RGB", (mosaic_w, mosaic_h), color="white")
    for i, img_path in enumerate(random_images):
        try:
            annotated = annotate_single_image(
                img_path=img_path,
                gt_colors=gt_colors,
                pred_colors=pred_colors,
                class_names=class_names,
                model=model,
                iou=iou,
                conf=conf,
                target_size=(tile_w, tile_h),
                line_width=line_width,
            )
            x_off = (i % cols) * tile_w
            y_off = (i // cols) * tile_h
            mosaic.paste(annotated, (x_off, y_off))
        except Exception as e:
            print(f"Could not process image {img_path}: {e}")

    # ---- Render final mosaic + legend with Matplotlib (no extra drawing) ----
    dpi = 200
    fig_w = mosaic_w / dpi
    fig_h = mosaic_h / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(mosaic)
    ax.set_axis_off()

    # Legend using clean Matplotlib handles
    handles = []
    for i, name in enumerate(class_names):
        if i < len(gt_colors):
            handles.append(Line2D([0], [0], color=gt_colors[i], lw=line_width, label=f"GT {name}"))
    for i, name in enumerate(class_names):
        if i < len(pred_colors):
            handles.append(Line2D([0], [0], color=pred_colors[i], lw=line_width, linestyle="--", label=f"Pred {name}"))

    if handles:
        ax.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=True,
            title="Legend",
        )

    plt.tight_layout()
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Mosaic saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Annotate each image (GT + predictions) using Matplotlib, then build a mosaic. Legend rendered with Matplotlib."
    )
    parser.add_argument("--list", help="Path to the text file containing the list of image filenames.")
    parser.add_argument("-n", type=int, default=9, help="Number of random images to use in the mosaic.")
    parser.add_argument("-o", "--output", default="mosaic.png", help="Output file for the mosaic image.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for image selection.")
    parser.add_argument("--image-dir", help="Parent directory for relative image paths.")
    parser.add_argument("--weights", type=str, default=None, help="Path to YOLO model weights (.pt). Runs inference per image if provided.")
    parser.add_argument("--iou", type=float, default=0.3, help="IOU threshold for inference.")
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold for inference.")
    parser.add_argument("--gt-colors", type=str, help="Comma-separated list of colors for ground truth boxes.")
    parser.add_argument("--pred-colors", type=str, help="Comma-separated list of colors for prediction boxes.")
    parser.add_argument("--class-names", type=str, help="Comma-separated list of class names for the legend.")
    parser.add_argument("--line-width", type=int, default=3, help="Line width for box outlines (Matplotlib).")

    args = parser.parse_args()

    image_files = []
    if args.list:
        if not os.path.exists(args.list):
            print(f"Error: File not found at {args.list}")
            return
        with open(args.list, "r") as f:
            image_files = [line.strip() for line in f.readlines() if line.strip()]
        if args.image_dir:
            image_files = [os.path.join(args.image_dir, fname) for fname in image_files]
    elif args.image_dir:
        if not os.path.isdir(args.image_dir):
            print(f"Error: Image directory not found at {args.image_dir}")
            return
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
        image_files = [
            os.path.join(args.image_dir, f)
            for f in os.listdir(args.image_dir)
            if os.path.splitext(f)[1].lower() in image_extensions
        ]
    else:
        print("Error: You must provide either --list or --image-dir.")
        return

    if not image_files:
        print("No image files found.")
        return

    gt_colors = args.gt_colors.split(",") if args.gt_colors else ["green", "red"]
    pred_colors = args.pred_colors.split(",") if args.pred_colors else ["blue", "orange"]
    class_names = args.class_names.split(",") if args.class_names else ["Shepherd's Crook", "Canker"]

    create_mosaic(
        image_files,
        args.n,
        args.output,
        seed=args.seed,
        weights_path=args.weights,
        iou=args.iou,
        conf=args.conf,
        gt_colors=gt_colors,
        pred_colors=pred_colors,
        class_names=class_names,
        line_width=args.line_width,
    )


if __name__ == "__main__":
    main()
