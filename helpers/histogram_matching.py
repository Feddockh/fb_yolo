import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms

def match_image_histogram(source_img, reference_img):
    matched = match_histograms(source_img, reference_img)
    return np.uint8(matched)

def color_transfer(source, target):
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    src_mean, src_std = cv2.meanStdDev(source_lab)
    tgt_mean, tgt_std = cv2.meanStdDev(target_lab)

    l, a, b = cv2.split(source_lab)
    l = ((l - src_mean[0][0]) * (tgt_std[0][0] / (src_std[0][0] + 1e-8))) + tgt_mean[0][0]
    a = ((a - src_mean[1][0]) * (tgt_std[1][0] / (src_std[1][0] + 1e-8))) + tgt_mean[1][0]
    b = ((b - src_mean[2][0]) * (tgt_std[2][0] / (src_std[2][0] + 1e-8))) + tgt_mean[2][0]

    transferred = cv2.merge([l, a, b])
    transferred = np.clip(transferred, 0, 255).astype("uint8")
    return cv2.cvtColor(transferred, cv2.COLOR_LAB2BGR)

source_img = cv2.imread('/home/hayden/cmu/kantor_lab/fb_auto_anno/test_imgs/IMG_2040.jpg')
reference_img = cv2.imread('/home/hayden/cmu/kantor_lab/fb_auto_anno/test_imgs/test.png')
matched_img = match_image_histogram(source_img, reference_img)

# Convert BGR to RGB for matplotlib
matched_img_rgb = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 6))
plt.imshow(matched_img_rgb)
plt.title('Matched Image')
plt.axis('off')
plt.show()

