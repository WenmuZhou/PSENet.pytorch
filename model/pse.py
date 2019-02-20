# -*- coding: utf-8 -*-
# @Time    : 1/26/19 5:59 PM
# @Author  : zhoujun
import time
import numpy as np
import cv2


def pse(kernels, min_area):
    kernel_num = len(kernels)  # 判断里面有几个kernel
    pred = np.zeros(kernels[0].shape, dtype=np.uint8)  # 最终的预测输出
    _, label, stats, centroids = cv2.connectedComponentsWithStats(kernels[0].astype(np.uint8), connectivity=4)
    for stat in stats[1:]:
        if stat[-1] < min_area:
            label[stat[1]:stat[1] + stat[3], stat[0]:stat[0] + stat[2]] = 0

    queue = []
    next_queue = []
    points = np.array(np.where(label > 0)).transpose((1, 0))  # points存储所有点坐标(x,y)

    for point_idx in range(points.shape[0]):  # 遍历每一个点
        x, y = points[point_idx, 0], points[point_idx, 1]  # 取出每一个点的坐标
        l = label[x, y]  # 取出点的label值
        queue.append((x, y, l))  # 将该点送入队列
        # queue.put((x, y, l))
        pred[x, y] = l  # 对该点赋值

    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    # for kernel_idx in range(1, kernel_num):  # 遍历每一个kernel
    for kernel in kernels[1:]:  # 遍历每一个kernel
        tic = time.time()
        while len(queue):
            (x, y, l) = queue.pop(0)  # 点出队列
            is_edge = True
            for j in range(4):  # 邻域判断
                tmpx = x + dx[j]  # 四个邻域遍历
                tmpy = y + dy[j]
                if tmpx < 0 or tmpx >= kernel.shape[0] or tmpy < 0 or tmpy >= kernel.shape[1]:
                    continue
                if kernel[tmpx, tmpy] == 0 or pred[tmpx, tmpy] > 0:
                    continue
                queue.append((tmpx, tmpy, l))
                pred[tmpx, tmpy] = l
                is_edge = False
            if is_edge:
                # 如果当前边界像素没有被替代，就更新下一个map的边界像素
                next_queue.append((x, y, l))
        # print(time.time()-tic)
        queue, next_queue = next_queue, queue
    return pred, len(stats[1:])


def decode(preds, threshold=0.5):
    # preds = (preds >= threshold).detach()
    # preds = (preds * preds[-1]).cpu().numpy()
    # np.save('result.npy', preds)
    # pred, label_num = pse(preds,100)

    mask = (preds[-1] > threshold).detach().float()
    preds = (preds * mask).detach().cpu().numpy()
    pred, label_num = pse(preds >= threshold, 100)
    h, w = pred.shape[-2:]
    bbox_list = []
    for label_idx in range(1, label_num + 1):
        result = (pred == label_idx).astype(np.uint8)
        _, contours, hierarchy = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            point = cv2.boxPoints(rect)
            point[:, 0] = np.clip(point[:, 0], 0, w - 1)
            point[:, 1] = np.clip(point[:, 1], 0, h - 1)
            bbox_list.append([point[1], point[2], point[3], point[0]])

    return pred, np.array(bbox_list)


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
    import time

    tic = time.time()
    pred = pse(kernels, 100)
    print(time.time() - tic)
