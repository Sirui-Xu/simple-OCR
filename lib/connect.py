import os
import sys
import cv2
import numpy as np
from skimage.measure import label, regionprops

# 计算连通域
def connect(img, *args, **kwargs):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(3,3),0)
    if np.mean(gray) < 128:
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    else:
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    labeled_img, num = label(binary, connectivity=1, background=0, return_num=True, *args, **kwargs)
    return regionprops(labeled_img)

# 生成边框
def connect_bbox(img, regions):
    vis = img.copy()
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        cv2.rectangle(vis, (minc, minr), (maxc, maxr), (255, 0, 0), 1)
    return vis

# 计算最大连通域
def largeConnectComponent(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(3,3),0)
    if np.mean(gray) < 128:
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    else:
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    labeled_img, num = label(binary, connectivity=1, background=0, return_num=True)
    regions = regionprops(labeled_img)
    bbox_area = [region.bbox_area for region in regions]
    bbox = regions[bbox_area.index(max(bbox_area))].bbox
    minr, minc, maxr, maxc = bbox
    return binary[minr:maxr, minc:maxc]


if __name__ == "__main__":
    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    regions = connect(img)
    output = connect_bbox(img, regions)
    cv2.imwrite('../results/connect.png', output)
    # cv2.waitKey(0)
