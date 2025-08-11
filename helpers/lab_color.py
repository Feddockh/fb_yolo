import cv2
import matplotlib.pyplot as plt

# Load image in RGB
image_bgr = cv2.imread('/home/hayden/cmu/kantor_lab/fb_auto_anno/test_imgs/IMG_2040.jpg')
# image_bgr = cv2.imread('/home/hayden/cmu/kantor_lab/fb_auto_anno/test_imgs/test.png')
# image_bgr = cv2.imread('/home/hayden/cmu/kantor_lab/fb_auto_anno/batch_1_new_annos/images/train/1739376932_848088832.png')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Convert to LAB
lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
l_channel, a_channel, b_channel = cv2.split(lab)

# Plot the LAB channels
plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(image_rgb)
plt.title('Original RGB')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(l_channel, cmap='gray')
plt.title('L (Lightness)')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(a_channel, cmap='RdYlGn')  # Red–Green
plt.title('A (Green–Red)')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(b_channel, cmap='coolwarm')  # Blue–Yellow
plt.title('B (Blue–Yellow)')
plt.axis('off')

plt.tight_layout()
plt.show()
