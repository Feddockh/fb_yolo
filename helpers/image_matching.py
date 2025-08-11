import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms


# Re-run with the updated argument for channel_axis instead of deprecated 'multichannel'
rgb_image = cv2.imread('/home/hayden/cmu/kantor_lab/fb_auto_anno/test_imgs/IMG_2040.jpg')
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
flash_image = cv2.imread('/home/hayden/cmu/kantor_lab/fb_auto_anno/batch_1_new_annos/images/train/1739374017_346715392.png')
flash_image = cv2.cvtColor(flash_image, cv2.COLOR_BGR2RGB)

matched_image = match_histograms(rgb_image, flash_image, channel_axis=-1)
matched_image = np.clip(matched_image, 0, 255).astype(np.uint8)

# Plot the results
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].imshow(rgb_image)
axs[0].set_title("Original RGB Image")
axs[0].axis("off")

axs[1].imshow(flash_image)
axs[1].set_title("Flash Reference Image")
axs[1].axis("off")

axs[2].imshow(matched_image)
axs[2].set_title("RGB Matched to Flash")
axs[2].axis("off")

plt.tight_layout()
plt.show()
