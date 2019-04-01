# -*- coding: utf-8 -*-
# @Time    : 2019/1/12 13:06

import cv2
import numbers
import math
import random
import numpy as np
from skimage.util import random_noise


def show_pic(img, bboxes=None, name='pic'):
    '''
    输入:
        img:图像array
        bboxes:图像的所有boudning box list, 格式为[[x_min, y_min, x_max, y_max]....]
        names:每个box对应的名称
    '''
    show_img = img.copy()
    if not isinstance(bboxes, np.ndarray):
        bboxes = np.array(bboxes)
    for point in bboxes.astype(np.int):
        cv2.line(show_img, tuple(point[0]), tuple(point[1]), (255, 0, 0), 2)
        cv2.line(show_img, tuple(point[1]), tuple(point[2]), (255, 0, 0), 2)
        cv2.line(show_img, tuple(point[2]), tuple(point[3]), (255, 0, 0), 2)
        cv2.line(show_img, tuple(point[3]), tuple(point[0]), (255, 0, 0), 2)
    # cv2.namedWindow(name, 0)  # 1表示原图
    # cv2.moveWindow(name, 0, 0)
    # cv2.resizeWindow(name, 1200, 800)  # 可视化的图片大小
    cv2.imshow(name, show_img)


# 图像均为cv2读取
class DataAugment():
    def __init__(self):
        pass

    def add_noise(self, im: np.ndarray):
        """
        对图片加噪声
        :param img: 图像array
        :return: 加噪声后的图像array,由于输出的像素是在[0,1]之间,所以得乘以255
        """
        return (random_noise(im, mode='gaussian', clip=True) * 255).astype(im.dtype)

    def random_scale(self, imgs: list, scales: np.ndarray or list, input_size: int) -> list:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param imgs: 原图 和 label
        :param scales: 尺度
        :param input_size: 图片短边的长度
        :return: 经过缩放的图片和文本
        """
        rd_scale = float(np.random.choice(scales))
        for idx in range(len(imgs)):
            imgs[idx] = cv2.resize(imgs[idx], dsize=None, fx=rd_scale, fy=rd_scale)
            imgs[idx], _ = self.rescale(imgs[idx], min_side=input_size)
        return imgs

    def rescale(self, img, min_side):
        h, w = img.shape[:2]
        scale = 1.0
        if min(h, w) < min_side:
            if h <= w:
                scale = 1.0 * min_side / h
            else:
                scale = 1.0 * min_side / w
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
        return img

    def random_rotate_img_bbox(self, imgs, degrees: numbers.Number or list or tuple or np.ndarray,
                               same_size=False):
        """
        从给定的角度中选择一个角度，对图片和文本框进行旋转
        :param imgs: 原图 和 label
        :param degrees: 角度，可以是一个数值或者list
        :param same_size: 是否保持和原图一样大
        :return: 旋转后的图片和角度
        """
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            degrees = (-degrees, degrees)
        elif isinstance(degrees, list) or isinstance(degrees, tuple) or isinstance(degrees, np.ndarray):
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            degrees = degrees
        else:
            raise Exception('degrees must in Number or list or tuple or np.ndarray')
        # ---------------------- 旋转图像 ----------------------
        w = imgs[0].shape[1]
        h = imgs[0].shape[0]
        angle = np.random.uniform(degrees[0], degrees[1])

        if same_size:
            nw = w
            nh = h
        else:
            # 角度变弧度
            rangle = np.deg2rad(angle)
            # 计算旋转之后图像的w, h
            nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w))
            nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w))
        # 构造仿射矩阵
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, 1)
        # 计算原图中心点到新图中心点的偏移量
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # 更新仿射矩阵
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        for idx in range(len(imgs)):
            # 仿射变换
            imgs[idx] = cv2.warpAffine(imgs[idx], rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
        return imgs

    def random_crop(self, imgs, img_size):
        h, w = imgs[0].shape[0:2]
        th, tw = img_size
        if w == tw and h == th:
            return imgs

        # label中存在文本实例，并且按照概率进行裁剪
        if np.max(imgs[1][:, :, -1]) > 0 and random.random() > 3.0 / 8.0:
            # 文本实例的top left点
            tl = np.min(np.where(imgs[1][:, :, -1] > 0), axis=1) - img_size
            tl[tl < 0] = 0
            # 文本实例的 bottom right 点
            br = np.max(np.where(imgs[1][:, :, -1] > 0), axis=1) - img_size
            br[br < 0] = 0
            # 保证选到右下角点是，有足够的距离进行crop
            br[0] = min(br[0], h - th)
            br[1] = min(br[1], w - tw)
            for _ in range(50000):
                i = random.randint(tl[0], br[0])
                j = random.randint(tl[1], br[1])
                # 保证最小的图有文本
                if imgs[1][:, :, 0][i:i + th, j:j + tw].sum() <= 0:
                    continue
                else:
                    break
        else:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)

        # return i, j, th, tw
        for idx in range(len(imgs)):
            if len(imgs[idx].shape) == 3:
                imgs[idx] = imgs[idx][i:i + th, j:j + tw, :]
            else:
                imgs[idx] = imgs[idx][i:i + th, j:j + tw]
        return imgs


    def horizontal_flip(self, imgs: list) -> list:
        """
        对图片和文本框进行水平翻转
        :param im: 图片
        :param text_polys: 文本框
        :return: 水平翻转之后的图片和文本框
        """
        for idx in range(len(imgs)):
            imgs[idx] = cv2.flip(imgs[idx], 1)
        return imgs

    def vertical_flip(self, imgs: list) -> list:
        """
         对图片和文本框进行竖直翻转
        :param im: 图片
        :param text_polys: 文本框
        :return: 竖直翻转之后的图片和文本框
        """
        for idx in range(len(imgs)):
            imgs[idx] = cv2.flip(imgs[idx], 0)
        return imgs