import cv2
import numpy as np
# 利用大津算法将图像二值化
def threshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) < 128:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    return binary