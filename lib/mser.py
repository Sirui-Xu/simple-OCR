import os
import sys
import cv2
import numpy as np
from PIL import Image

def mser(img, *args, **kwargs):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create(*args, **kwargs)
    regions, boxes = mser.detectRegions(gray)
    return regions, boxes

# 生成边框
def mser_bbox(img, boxes):
    vis = img.copy()
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 1)
    return vis

# 生成凸包框
def mser_hull(img, regions):
    vis = img.copy()
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(vis, hulls, 1, (0, 255, 0))
    return vis

# 利用mask只保留文字部分
def mser_text(img, regions):
    vis = img.copy()
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
    for contour in hulls:
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
    text_only = cv2.bitwise_and(vis, vis, mask=mask)
    return text_only


if __name__=='__main__':
    file_path = sys.argv[1]
    img = cv2.imread(file_path)
    regions, boxes = mser(img, _min_area=50)
    img_box = mser_bbox(img, boxes)
    img_hull = mser_hull(img, regions)
    img_text = mser_text(img, regions)
    output = np.hstack([img_box, img_hull, img_text])
    cv2.imwrite('../results/mser_f.png', output)