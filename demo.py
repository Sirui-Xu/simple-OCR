import os
import sys
import numpy
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from pipeline import train_svm, pipeline_plus, boost

import matplotlib.pyplot as plt
import numpy as np

char_data = [chr(i) for i in range(65,91)] + [chr(i) for i in range(97,123)]
img_path = sys.argv[1]
data_path = './data/'
img = cv2.imread(img_path)
boxes, labels = pipeline_plus(img, data_path)


# 鼠标移动的事件，当移动到检测出文字的部分，则画出边框和识别出的字母。
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    vis = img.copy()
    if event == cv2.EVENT_MOUSEMOVE:
        for i, box in enumerate(boxes):
            x_, y_, w_, h_ = box
            # 当鼠标的坐标在边框内，则输出
            if x >= x_ and x <= x_ + w_ and y >= y_ and y <= y_ + h_:
                print(f'({x_}, {y_}) to ({x_+w_}, {y_+h_}) is ' + char_data[labels[i]])
                cv2.rectangle(vis, (x_, y_), (x_ + w_, y_ + h_), (0, 0, 255), 1)
                cv2.putText(vis, char_data[labels[i]], (x_ + w_, y_ + h_), cv2.FONT_HERSHEY_PLAIN,
                            4.0, (0, 0, 255), thickness=2)
                cv2.imshow("image", vis)


cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", img)

# 鼠标移动时不退出
while (True):
    try:
        cv2.waitKey(100)
    except Exception:
        cv2.destroyAllWindows()
        break

cv2.waitKey(0)
cv2.destroyAllWindows()