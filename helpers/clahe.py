import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_clahe(image_rgb):
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # More aggressive CLAHE
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4, 4))
    cl = clahe.apply(l)
    
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def adjust_gamma(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def boost_saturation(image_rgb, factor=1.2):
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

def invert_brightness(image_rgb, scale=0.5):
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv_inverted = hsv.copy()
    hsv_inverted[:, :, 2] = 255 - hsv_inverted[:, :, 2]
    inverted_image = cv2.cvtColor(hsv_inverted.astype(np.uint8), cv2.COLOR_HSV2RGB)
    # Blend the original and inverted images based on the scale factor.
    # scale=0.5 means equal blending. Adjust scale as needed.
    return cv2.addWeighted(image_rgb, scale, inverted_image, 1 - scale, 0)

# Load in RGB format
img = cv2.cvtColor(cv2.imread('/home/hayden/cmu/kantor_lab/fb_auto_anno/batch_1_new_annos/images/train/1739373919_100068096.png'), cv2.COLOR_BGR2RGB)
clahe_img = apply_clahe(img)
clahe_img = adjust_gamma(clahe_img, gamma=2.0)
clahe_img = invert_brightness(clahe_img, scale=0.5)



plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(clahe_img)
plt.title('CLAHE Image')
plt.axis('off')

plt.tight_layout()
plt.show()