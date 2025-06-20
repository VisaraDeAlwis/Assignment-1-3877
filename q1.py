import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("0030_01.jpg", cv2.IMREAD_GRAYSCALE)

Quantization_Level_list = [256, 128, 64, 32, 16, 8, 4, 2]

plt.figure(figsize=(16, 10))

for idx, level in enumerate(Quantization_Level_list):
    step = 256 // level
    quantized = (image // step) * step

    plt.subplot(2, 4, idx + 1)
    plt.imshow(quantized, cmap='gray') 
    plt.title(f'{level} Levels',fontsize=20)
    plt.axis('off')

plt.show()
