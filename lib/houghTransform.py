import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def houghTransform(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯模糊
    gaus = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(gaus, 50, 200)
    #hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=60, maxLineGap=10)
    lines1 = lines[:,0,:]   # 提取为二维
    return lines1


if __name__ == "__main__":
    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    lines = houghTransform(img)
    for x1,y1,x2,y2 in lines[:]: 
        cv2.line(img, (x1,y1), (x2,y2), (0, 255, 0), 1)
    cv2.imwrite('../results/houghTransform.png', img)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)