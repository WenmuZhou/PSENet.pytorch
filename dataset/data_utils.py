# -*- coding: utf-8 -*-
# @Time    : 2018/6/11 15:54
# @Author  : zhoujun

import os
import random
import pathlib
import pyclipper
from torch.utils import data
import glob
import numpy as np
import cv2
from dataset.augment import DataAugment
from utils.utils import draw_bbox

data_aug = DataAugment()


def check_and_validate_polys(polys, xxx_todo_changeme):
    '''
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    :param polys:
    :param tags:
    :return:
    '''
    (h, w) = xxx_todo_changeme
    if polys.shape[0] == 0:
        return polys
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)  # x coord not max w-1, and not min 0
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)  # y coord not max h-1, and not min 0

    validated_polys = []
    for poly in polys:
        p_area = cv2.contourArea(poly)
        if abs(p_area) < 1:
            continue
        validated_polys.append(poly)
    return np.array(validated_polys)


def generate_rbox(im_size, text_polys, text_tags, training_mask, i, n, m):
    """
    生成mask图，白色部分是文本，黑色是北京
    :param im_size: 图像的h,w
    :param text_polys: 框的坐标
    :param text_tags: 标注文本框是否参与训练
    :return: 生成的mask图
    """
    h, w = im_size
    score_map = np.zeros((h, w), dtype=np.uint8)
    for poly, tag in zip(text_polys, text_tags):
        poly = poly.astype(np.int)
        r_i = 1 - (1 - m) * (n - i) / (n - 1)
        d_i = cv2.contourArea(poly) * (1 - r_i * r_i) / cv2.arcLength(poly, True)
        pco = pyclipper.PyclipperOffset()
        # pco.AddPath(pyclipper.scale_to_clipper(poly), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        # shrinked_poly = np.floor(np.array(pyclipper.scale_from_clipper(pco.Execute(-d_i)))).astype(np.int)
        pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        shrinked_poly = np.array(pco.Execute(-d_i))
        cv2.fillPoly(score_map, shrinked_poly, 1)
        # 制作mask
        # rect = cv2.minAreaRect(shrinked_poly)
        # poly_h, poly_w = rect[1]

        # if min(poly_h, poly_w) < 10:
        #     cv2.fillPoly(training_mask, shrinked_poly, 0)
        if tag:
            cv2.fillPoly(training_mask, shrinked_poly, 0)
        # 闭运算填充内部小框
        # kernel = np.ones((3, 3), np.uint8)
        # score_map = cv2.morphologyEx(score_map, cv2.MORPH_CLOSE, kernel)
    return score_map, training_mask


def augmentation(im: np.ndarray, text_polys: np.ndarray, scales: np.ndarray, degrees: int, input_size: int) -> tuple:
    # the images are rescaled with ratio {0.5, 1.0, 2.0, 3.0} randomly
    im, text_polys = data_aug.random_scale(im, text_polys, scales)
    # the images are horizontally fliped and rotated in range [−10◦, 10◦] randomly
    if random.random() < 0.5:
        im, text_polys = data_aug.horizontal_flip(im, text_polys)
    if random.random() < 0.5:
        im, text_polys = data_aug.random_rotate_img_bbox(im, text_polys, degrees)
    # 640 × 640 random samples are cropped from the transformed images
    # im, text_polys = data_aug.random_crop_img_bboxes(im, text_polys)

    # im, text_polys = data_aug.resize(im, text_polys, input_size, keep_ratio=False)
    # im, text_polys = data_aug.random_crop_image_pse(im, text_polys, input_size)

    return im, text_polys


def image_label(im_fn: str, text_polys: np.ndarray, text_tags: list, n: int, m: float, input_size: int,
                defrees: int = 10,
                scales: np.ndarray = np.array([0.5, 1, 2.0, 3.0])) -> tuple:
    '''
    get image's corresponding matrix and ground truth
    return
    images [512, 512, 3]
    score  [128, 128, 1]
    geo    [128, 128, 5]
    mask   [128, 128, 1]
    '''

    im = cv2.imread(im_fn)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    h, w, _ = im.shape
    # 检查越界
    text_polys = check_and_validate_polys(text_polys, (h, w))
    im, text_polys, = augmentation(im, text_polys, scales, defrees, input_size)

    h, w, _ = im.shape
    short_edge = min(h, w)
    if short_edge < input_size:
        # 保证短边 >= inputsize
        scale = input_size / short_edge
        im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
        text_polys *= scale

    # # normal images
    # im = im.astype(np.float32)
    # im /= 255.0
    # im -= np.array((0.485, 0.456, 0.406))
    # im /= np.array((0.229, 0.224, 0.225))

    h, w, _ = im.shape
    training_mask = np.ones((h, w), dtype=np.uint8)
    score_maps = []
    for i in range(1, n + 1):
        # s1->sn,由小到大
        score_map, training_mask = generate_rbox((h, w), text_polys, text_tags, training_mask, i, n, m)
        score_maps.append(score_map)
    score_maps = np.array(score_maps, dtype=np.float32)
    imgs = data_aug.random_crop_author([im, score_maps.transpose((1, 2, 0)),training_mask], (input_size, input_size))
    return imgs[0], imgs[1].transpose((2, 0, 1)), imgs[2]#im,score_maps,training_mask#


class MyDataset(data.Dataset):
    def __init__(self, data_dir, data_shape: int = 640, n=6, m=0.5, transform=None, target_transform=None):
        self.data_list = self.load_data(data_dir)
        self.data_shape = data_shape
        self.transform = transform
        self.target_transform = target_transform
        self.n = n
        self.m = m

    def __getitem__(self, index):
        # print(self.image_list[index])
        img_path, text_polys, text_tags = self.data_list[index]
        img, score_maps, training_mask = image_label(img_path, text_polys, text_tags, input_size=self.data_shape,
                                                     n=self.n,
                                                     m=self.m)
        # img = draw_bbox(img,text_polys)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            score_maps = self.target_transform(score_maps)
            training_mask = self.target_transform(training_mask)
        return img, score_maps, training_mask

    def load_data(self, data_dir: str) -> list:
        data_list = []
        for x in glob.glob(data_dir + '/img/*.jpg', recursive=True):
            d = pathlib.Path(x)
            label_path = os.path.join(data_dir, 'gt', (str(d.stem) + '.txt'))
            bboxs, text = self._get_annotation(label_path)
            if len(bboxs) > 0:
                data_list.append((x, bboxs, text))
            else:
                print('there is no suit bbox on {}'.format(label_path))
        return data_list[:10]

    def _get_annotation(self, label_path: str) -> tuple:
        boxes = []
        text_tags = []
        with open(label_path, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                try:
                    label = params[8]
                    if label == '*' or label == '###':
                        text_tags.append(True)
                    else:
                        text_tags.append(False)
                    # if label == '*' or label == '###':
                    x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, params[:8]))
                    boxes.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                except:
                    print('load label failed on {}'.format(label_path))
        return np.array(boxes, dtype=np.float32), np.array(text_tags, dtype=np.bool)

    def __len__(self):
        return len(self.data_list)

    def save_label(self, img_path, label):
        save_path = img_path.replace('img', 'save')
        if not os.path.exists(os.path.split(save_path)[0]):
            os.makedirs(os.path.split(save_path)[0])
        img = draw_bbox(img_path, label)
        cv2.imwrite(save_path, img)
        return img


if __name__ == '__main__':
    import torch
    import config
    from utils.utils import show_img
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from torchvision import transforms

    train_data = MyDataset(config.trainroot, data_shape=config.data_shape, n=config.n, m=config.m,
                           transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False, num_workers=0)

    pbar = tqdm(total=len(train_loader))
    for i, (img, label, mask) in enumerate(train_loader):
        print(label.shape)
        print(img.shape)
        print(label[0][-1].sum())
        print(mask[0].shape)
        # pbar.update(1)
        show_img((img[0] * mask[0].to(torch.float)).numpy().transpose(1, 2, 0), color=True)
        show_img(label[0])
        show_img(mask[0])
        plt.show()

    pbar.close()
