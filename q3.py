import cv2
import matplotlib.pyplot as plt

img_bgr = cv2.imread('0030_01.jpg')

img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

(h, w) = img.shape[:2]
center = (w // 2, h // 2)

M_45 = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated_45 = cv2.warpAffine(img, M_45, (w, h))

rotated_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

images = [img, rotated_45, rotated_90]
titles = ['Original', 'Rotated 45°', 'Rotated 90°']

plt.figure(figsize=(12, 6))
for i in range(len(images)):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i])  
    plt.title(titles[i],fontsize=14)
    plt.axis('off')

plt.tight_layout()
plt.show()
