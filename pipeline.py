import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import label, regionprops
from lib.mser import mser, mser_bbox
from lib.connect import connect, connect_bbox, largeConnectComponent
from lib.utils import threshold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import svm

char_data = [chr(i) for i in range(65,91)] + [chr(i) for i in range(97,123)]
# 自己实现的简易版非极大抑制
def non_max_suppression(dets, probs, overlapThresh=0.15):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2] + x1
    y2 = dets[:, 3] + y1
    scores = probs

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    # score从大到小，排除那些重合程度高的box（通过IoU 交并比）
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= overlapThresh)[0]
        order = order[inds + 1]

    return dets[keep]
# print(char_data)
# 集成mser和连通域的算法
def boost(img, multi=True):
    # 调用连通域算法
    connect_regions = connect(img)
    connect_area = [region.area for region in connect_regions]
    min_area = max(15, min(connect_area))
    max_area = max(connect_area)
    # 调用mser算法
    mser_regions, bbox = mser(img, _min_area=min_area, _max_area=max_area)
    mser_area = [region.shape[0] for region in mser_regions]
    # bbox_area = [w * h for x, y, w, h in bbox]
    # bbox_area.extend(connect_bbox_area)
    # bbox = list(bbox)
    connect_area = np.array(connect_area)
    # bbox.extend([(region.bbox[1], region.bbox[0], region.bbox[3] - region.bbox[1], region.bbox[2] - region.bbox[0]) for region in connect_regions])
    # bbox_area = np.array(bbox_area).reshape(-1, 1)
    # print(connect_bbox_area)
    max_area = max(mser_area)
    # box面积越大 可信度越高。
    confidences = [area / max_area for area in mser_area]
    '''
    if multi:
        Scores = []  # 存放轮廓系数
        for k in range(2, 5):
            estimator = KMeans(n_clusters=k)  # 构造聚类器
            estimator.fit(bbox_area)
            Scores.append(silhouette_score((bbox_area), estimator.labels_, metric='euclidean'))
        k = Scores.index(max(Scores)) + 2
        estimator = KMeans(n_clusters=k)
        estimator.fit(bbox_area)
        confidences = [np.sum((1 / (1 + np.abs(estimator.cluster_centers_ - value))).flatten())/ value[0] for value in bbox_area]
    else:
        confidences = [np.sum(np.exp(-np.abs(np.mean(bbox_area) - value))) for value in bbox_area]
    '''
    confidences = np.array(confidences)
    # print(confidences)
    # print(bbox_area[0])
    boxes = non_max_suppression(np.array(bbox), probs=confidences, overlapThresh=0.1)
    # boxes = [(region.bbox[1], region.bbox[0], region.bbox[3] - region.bbox[1], region.bbox[2] - region.bbox[0]) for region in connect_regions]
    # return [(region.bbox[1], region.bbox[0], region.bbox[3] - region.bbox[1], region.bbox[2] - region.bbox[0]) for region in connect_regions]
    return boxes

# 加载数据集，用于sift算子
def dataloader(data_path):
    data = {}
    char_imgs = os.listdir(data_path)
    detect = cv2.xfeatures2d.SIFT_create()
    for char_img in char_imgs:
        # if char_img[0] not in char_data:
        #     continue
        char = cv2.imread(os.path.join(data_path, char_img))
        char_name = char_img.split('-')[0]
        lcc = largeConnectComponent(char)
        # cv2.imshow('lcc', lcc)
        # cv2.waitKey(0)
        _, des = detect.detectAndCompute(lcc,None)
        if char_name not in data:
            data[char_name] = [des]
        else:
            data[char_name].append(des)
    return data

# 加载数据集直接逐像素特征（将2维图像直接压缩成1维特征）
def imageloader(data_path):
    data = {}
    char_imgs = os.listdir(data_path)
    for char_img in char_imgs:
        # if char_img[0] not in char_data:
        #     continue
        char = cv2.imread(os.path.join(data_path, char_img))
        char_name = chr(int(char_img.split('-')[0]))
        lcc = largeConnectComponent(char)
        lcc = cv2.resize(lcc, (15, 20), interpolation=cv2.INTER_CUBIC)
        lcc[lcc < 128] = 0
        lcc[lcc >= 128] = 1
        if char_name not in data:
            # numpy直接将二维向量压成一维向量
            data[char_name] = [lcc.flatten()]
        else:
            data[char_name].append(lcc.flatten())
    return data

# 计算brute-force匹配分数
def match_score(des1, des2):
    bf = cv2.BFMatcher()
    score = 0
    matches = bf.knnMatch(des1, des2, k=2)
    # print(matches)
    for match in matches:
        if len(match) == 2:
            m, n = match
        else:
            # print('wrong!', match)
            continue
        # 匹配距离满足关系
        if m.distance < 0.75*n.distance:
            score += 1
    return score

# 利用sift算子匹配的算法，效果不好
def pipeline(img, data_path, thres=4):
    boxes = boost(img)
    fined_boxes = []
    features = dataloader(data_path)
    detect = cv2.xfeatures2d.SIFT_create()
    img = threshold(img)
    for box in boxes:
        x, y, w, h = box
        bw_pixels = img[y:y+h, x:x+w]
        # cv2.imshow('char', bw_pixels)
        # cv2.waitKey(0)
        img_des = []
        max_score = 0
        max_score_char = '.'
        for i in range(4):
            _, des = detect.detectAndCompute(np.rot90(bw_pixels, i), None)
            score = 0
            if des is None:
                print('this char has no feature')
                # cv2.imshow('error', bw_pixels)
                # cv2.waitKey(0)
                continue
            for char, char_des_list in features.items():
                for char_des in char_des_list:
                    if char_des is None:
                        continue
                    score += match_score(des, char_des)
                if score > max_score:
                    max_score = score
                    max_score_char = char

        if max_score >= thres and max_score_char in char_data:
            fined_boxes.append(box)
            cv2.imshow(char, bw_pixels)
            cv2.waitKey(0)
    return fined_boxes

# 利用SVM进行分类的算法 （训练分类器）
def train_svm(data_path):
    char_images = imageloader(data_path)
    X = []
    y = []
    for char, char_img_list in char_images.items():
        for char_feature in char_img_list:
            X.append(char_feature)
            if char in char_data:
                y.append(char_data.index(char))
            else:
                raise Exception('unknown word')
    
    clf=svm.SVC()
    clf.fit(X, y)
    return clf

# 更新后的pipeline，直接使用逐像素特征。
def pipeline_plus(img, data_path):
    boxes = boost(img)
    fined_boxes = []
    clf = train_svm(data_path)

    img = threshold(img)
    labels = []
    for box in boxes:
        x, y, w, h = box
        bw_pixels = img[y:y+h, x:x+w]
        # cv2.imshow('char', bw_pixels)
        # cv2.waitKey(0)

        # 将字母图片统一拉成同样的大小进行比较
        bw = cv2.resize(bw_pixels, (15, 20), interpolation=cv2.INTER_CUBIC)
        bw[bw < 128] = 0
        bw[bw >= 128] = 1
        bw_feature = bw.flatten().reshape(1, -1)
        y_ = clf.predict(bw_feature)[0]

        labels.append(y_)

    return boxes, labels

if __name__ == "__main__":
    img_path = sys.argv[1]
    data_path = './data/'
    img = cv2.imread(img_path)
    boxes, labels = pipeline_plus(img, data_path)
    output = mser_bbox(img, boxes)
    for i, box in enumerate(boxes):
        x, y, w, h = box
        print(f'({x}, {y}) to ({x+w}, {y+h}) is ' + char_data[labels[i]])
        # 展示被检测出的字母
        bw_pixels = img[y:y+h, x:x+w]
        cv2.imshow(char_data[labels[i]], bw_pixels)
        cv2.waitKey(0)
    cv2.imwrite('./results/final.png', output)