# -*- coding: utf-8 -*-
# @Time    : 1/26/19 5:59 PM
# @Author  : zhoujun
import time
from utils import exe_time
from pse import pse
import numpy as np
import torch
import cv2


def decode_sigmoid(preds, scale, threshold=0.7311):
    """
    在输出上使用sigmoid 将值转换为置信度，并使用阈值来进行文字和背景的区分
    :param preds: 网络输出
    :param scale: 网络的scale
    :param threshold: sigmoid的阈值
    :return: 最后的输出图和文本框
    """
    preds = torch.sigmoid(preds)
    preds = preds.detach().cpu().numpy()

    score = preds[-1].astype(np.float32)
    preds = preds > threshold
    # preds = preds * preds[-1] # 使用最大的kernel作为其他小图的mask,不使用的话效果更好
    pred, label_values = pse(preds, 5)
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


def decode(preds, scale):
    """
    将输出值在1以下的作为背景，1以上的作为文字，使用 https://github.com/liuheng92/tensorflow_PSENet/blob/feature_dev/pse的合并算法
    :param preds: 网络输出
    :param scale: 网络的scale
    :return: 最后的输出图和文本框
    """
    score = torch.sigmoid(preds[-1])
    outputs = (torch.sign(preds - 1) + 1) // 2

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
    """
    将输出值在1以下的作为背景，1以上的作为文字，使用作者的合并算法
    :param preds: 网络输出
    :param scale: 网络的scale
    :return: 最后的输出图和文本框
    """
    from author_pse import pse as apse

    score = torch.sigmoid(preds[-1])
    outputs = (torch.sign(preds - 1) + 1) // 2

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
