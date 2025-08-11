#!/usr/bin/env python3
import os
import cv2
from tqdm import tqdm

# ─── Configuration ─────────────────────────────────────────────────────────────

IMAGE_DIR = "batch_1_new_annos/images/train"
LABEL_DIR = "batch_1_new_annos/labels/train"
OUTPUT_DIR = "batch_1_new_annos/annotated_images/train"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List of class names, in order of their class IDs
CLASS_NAMES = [
    "Shepherd's Crook",
    "Stem End Canker",
    "Elliptical Canker",
    "Girdling Canker",
    "Blossom Blight",
    "Rootstock Blight",
    "Stem End Canker 2",
]

# Hex → BGR converter
def hex_to_bgr(h):
    h = h.lstrip('#')
    r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)

# Map class name → BGR color
LABEL_COLOR_MAP = {
    "Shepherd's Crook":   hex_to_bgr("#3df53d"),
    "Stem End Canker":    hex_to_bgr("#33ddff"),
    "Elliptical Canker":  hex_to_bgr("#ffcc33"),
    "Girdling Canker":    hex_to_bgr("#fa3253"),
    "Blossom Blight":     hex_to_bgr("#fa32b7"),
    "Rootstock Blight":   hex_to_bgr("#f59331"),
    "Stem End Canker 2":  hex_to_bgr("#b83df5"),
}

# Drawing settings
FONT              = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE        = 0.6
FONT_THICKNESS    = 2
BOX_THICKNESS     = 2

# ─── Helpers ───────────────────────────────────────────────────────────────────

def yolo_to_bbox(xc, yc, w, h, img_w, img_h):
    xc, yc, w, h = xc*img_w, yc*img_h, w*img_w, h*img_h
    x1 = int(xc - w/2); y1 = int(yc - h/2)
    x2 = int(xc + w/2); y2 = int(yc + h/2)
    return x1, y1, x2, y2

# ─── Main ─────────────────────────────────────────────────────────────────────

def annotate_dataset():
    imgs = [f for f in os.listdir(IMAGE_DIR)
            if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]
    for img_name in tqdm(imgs, desc="Annotating"):
        img_path   = os.path.join(IMAGE_DIR, img_name)
        label_path = os.path.join(LABEL_DIR, os.path.splitext(img_name)[0] + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️  Could not read {img_name}, skipping.")
            continue
        h, w = img.shape[:2]

        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    cls_id, xc, yc, bw, bh = parts
                    cls_id = int(cls_id)
                    if not (0 <= cls_id < len(CLASS_NAMES)):
                        continue

                    name  = CLASS_NAMES[cls_id]
                    color = LABEL_COLOR_MAP[name]
                    x1,y1,x2,y2 = yolo_to_bbox(
                        float(xc), float(yc), float(bw), float(bh), w, h
                    )

                    # draw box
                    cv2.rectangle(img, (x1,y1), (x2,y2), color, BOX_THICKNESS)

                    # text BG
                    (tw, th), _ = cv2.getTextSize(name, FONT, FONT_SCALE, FONT_THICKNESS)
                    cv2.rectangle(
                        img,
                        (x1, y1 - th - 4),
                        (x1 + tw + 4, y1),
                        color,
                        cv2.FILLED
                    )
                    # label
                    cv2.putText(
                        img, name, (x1 + 2, y1 - 2),
                        FONT, FONT_SCALE, (255,255,255),
                        FONT_THICKNESS, cv2.LINE_AA
                    )

        # save
        out_path = os.path.join(OUTPUT_DIR, img_name)
        cv2.imwrite(out_path, img)

if __name__ == "__main__":
    annotate_dataset()
    print("✅ Done! Annotated images are in:", OUTPUT_DIR)