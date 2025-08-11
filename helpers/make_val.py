#!/usr/bin/env python3
import os
import random
import shutil

# ─── CONFIG ────────────────────────────────────────────────────────────────────
VAL_FRACTION = 0.2  # fraction of train images to move into val
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}  # image file extensions
ROOT = os.getcwd() # Run this code from dataset root folder
IMG_DIR = os.path.join(ROOT, "images")
LBL_DIR = os.path.join(ROOT, "labels")

TRAIN_IMG_DIR = os.path.join(IMG_DIR, "train")
VAL_IMG_DIR   = os.path.join(IMG_DIR, "val")
TRAIN_LBL_DIR = os.path.join(LBL_DIR, "train")
VAL_LBL_DIR   = os.path.join(LBL_DIR, "val")

TRAIN_LIST = os.path.join(ROOT, "train.txt")
VAL_LIST   = os.path.join(ROOT, "val.txt")
# ────────────────────────────────────────────────────────────────────────────────

def ensure_dirs():
    os.makedirs(VAL_IMG_DIR, exist_ok=True)
    os.makedirs(VAL_LBL_DIR, exist_ok=True)

def list_images(path):
    return [f for f in os.listdir(path)
            if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS]

def main():
    ensure_dirs()

    # 1. Gather all train images
    all_imgs = list_images(TRAIN_IMG_DIR)
    num_val = int(len(all_imgs) * VAL_FRACTION)
    val_imgs = set(random.sample(all_imgs, num_val))

    # 2. Move sampled images & labels into val dirs
    for img in val_imgs:
        src_img = os.path.join(TRAIN_IMG_DIR, img)
        dst_img = os.path.join(VAL_IMG_DIR, img)
        shutil.move(src_img, dst_img)

        lbl = os.path.splitext(img)[0] + ".txt"
        src_lbl = os.path.join(TRAIN_LBL_DIR, lbl)
        dst_lbl = os.path.join(VAL_LBL_DIR, lbl)
        if os.path.exists(src_lbl):
            shutil.move(src_lbl, dst_lbl)

    # 3. Rewrite train.txt & val.txt
    with open(TRAIN_LIST, "w") as f_train, open(VAL_LIST, "w") as f_val:
        # remaining train images
        for img in sorted(list_images(TRAIN_IMG_DIR)):
            rel = os.path.join("images/train", img)
            f_train.write(rel + "\n")
        # val images
        for img in sorted(list_images(VAL_IMG_DIR)):
            rel = os.path.join("images/val", img)
            f_val.write(rel + "\n")

    print(f"Split {num_val} images into validation.")
    print(f"train.txt → {len(list_images(TRAIN_IMG_DIR))} images")
    print(f"val.txt   → {len(list_images(VAL_IMG_DIR))} images")

if __name__ == "__main__":
    main()
