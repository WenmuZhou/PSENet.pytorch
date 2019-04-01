# -*- coding: utf-8 -*-
# @Time    : 1/26/19 5:59 PM
# @Author  : zhoujun
import time
from utils import exe_time
from pse import pse
import numpy as np
import torch
import cv2

def decode(preds, scale):
    score = torch.sigmoid(preds[-1])
    outputs = (torch.sign(preds - 1) + 1) / 2

    text = outputs[-1]
    kernels = outputs * text
    score = score.detach().cpu().numpy().astype(np.float32)
    kernels = kernels.detach().cpu().numpy()
    pred, label_values = pse(kernels.astype(np.uint8), 5 / (scale * scale))
    bbox_list = []
    for label_value in label_values:
        points = np.array(np.where(pred == label_value)).transpose((1, 0))[:, ::-1]

        if points.shape[0] < 800 / (scale * scale):
            continue

        score_i = np.mean(score[pred == label_value])
        if score_i < 0.93:
            continue

        rect = cv2.minAreaRect(points)
        bbox = cv2.boxPoints(rect)
        bbox_list.append([bbox[1], bbox[2], bbox[3], bbox[0]])

    return pred, np.array(bbox_list)


def decode_author(preds, scale):
    from author_pse import pse as apse

    score = torch.sigmoid(preds[-1])
    outputs = (torch.sign(preds - 1) + 1) / 2

    text = outputs[-1]
    kernels = outputs * text

    score = score.data.cpu().numpy().astype(np.float32)
    kernels = kernels.data.cpu().numpy().astype(np.uint8)

    # c++ version pse
    pred = apse(kernels, 5.0 / (scale * scale))
    # python version pse
    # pred = pypse(kernels, args.min_kernel_area / (args.scale * args.scale))

    label = pred
    label_num = np.max(label) + 1
    bboxes = []
    for i in range(1, label_num):
        points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]

        if points.shape[0] < 800 / (scale * scale):
            continue

        score_i = np.mean(score[label == i])
        if score_i < 0.93:
            continue

        rect = cv2.minAreaRect(points)
        bbox = cv2.boxPoints(rect)
        bboxes.append([bbox[1], bbox[2], bbox[3], bbox[0]])
        # bboxes.append(bbox.reshape(-1))
    return pred, np.array(bboxes)


if __name__ == '__main__':
    x = np.zeros((3, 3, 3))
    y = np.ones((3, 3, 3))
    s1 = np.zeros((5, 5))
    s2 = np.zeros((5, 5))
    s3 = np.zeros((5, 5))
    s1[[0, 0, 0, 0], [0, 1, 2, 3]] = 1
    s2[[2, 2, 2, 3, 3, 3], [0, 1, 2, 0, 1, 2]] = 1
    s3[[1, 1, 1, 1], [0, 1, 2, 3]] = 1
    # com = np.concatenate((x,y,x,y),axis=2)
    # kernels = np.stack((s1, s2, s3)).astype(np.uint8)
    kernels = np.load('/data1/zj/PSENet.pytorch/result.npy')

    tic = time.time()
    pred = pse(kernels, 100)
    print(time.time() - tic)
