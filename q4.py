import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

img_bgr = cv2.imread('0030_01.jpg')
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

output_dir = "block_average_outputs"
os.makedirs(output_dir, exist_ok=True)

def block_average(img, block_size):
    h, w = img.shape[:2]
    h_crop = h - (h % block_size)
    w_crop = w - (w % block_size)
    img_cropped = img[:h_crop, :w_crop]
    output = np.zeros_like(img_cropped)

    for i in range(0, h_crop, block_size):
        for j in range(0, w_crop, block_size):
            block = img_cropped[i:i+block_size, j:j+block_size]
            avg_color = np.mean(block, axis=(0, 1), dtype=int)
            output[i:i+block_size, j:j+block_size] = avg_color

    return output

block_sizes = [1, 3, 5, 7]
titles = ['Original', '3x3 Block Average', '5x5 Block Average', '7x7 Block Average']


for idx, bsize in enumerate(block_sizes):
    avg_img = block_average(img, bsize)

    
    filename = f"block_average_{bsize}x{bsize}.jpg"
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, cv2.cvtColor(avg_img, cv2.COLOR_RGB2BGR))

    
    plt.figure(figsize=(4, 4))
    plt.imshow(avg_img)
    plt.title(titles[idx])
    plt.axis('off')
    plt.show()
