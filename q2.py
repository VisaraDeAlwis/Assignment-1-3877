import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("0030_01.jpg", cv2.IMREAD_GRAYSCALE)

def average_filter(img, ksize):
    kernel = np.ones((ksize, ksize), dtype=np.float32) / (ksize * ksize) 
    filtered = cv2.filter2D(img, -1, kernel)
    return filtered

Avg_filter_sizes = [3, 10, 20]

plt.figure(figsize=(15, 6))

for index, fil_size in enumerate(Avg_filter_sizes):
    avg = average_filter(image, fil_size)
    plt.subplot(1, 3, index + 1)
    plt.imshow(avg, cmap='gray')
    plt.title(f"{fil_size}x{fil_size} Average Filter", fontsize=16)
    plt.axis('off')

plt.tight_layout()
plt.show()
