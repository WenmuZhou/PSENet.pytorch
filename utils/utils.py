# -*- coding: utf-8 -*-
# @Time    : 1/4/19 11:18 AM
# @Author  : zhoujun
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


def show_img(imgs: np.ndarray, color=False):
    if (len(imgs.shape) == 3 and color) or (len(imgs.shape) == 2 and not color):
        imgs = np.expand_dims(imgs, axis=0)
    for img in imgs:
        plt.figure()
        plt.imshow(img, cmap=None if color else 'gray')


def draw_bbox(img_path, result,color=(255, 0, 0)):
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
    img_path = img_path.copy()
    for point in result:
        cv2.line(img_path, tuple(point[0]), tuple(point[1]), color, 2)
        cv2.line(img_path, tuple(point[1]), tuple(point[2]), color, 2)
        cv2.line(img_path, tuple(point[2]), tuple(point[3]), color, 2)
        cv2.line(img_path, tuple(point[3]), tuple(point[0]), color, 2)
    return img_path


def setup_logger(log_file_path: str = None):
    import logging
    from colorlog import ColoredFormatter
    logging.basicConfig(filename=log_file_path, format='%(asctime)s %(levelname)-8s %(filename)s: %(message)s',
                        # 定义输出log的格式
                        datefmt='%Y-%m-%d %H:%M:%S', )
    """Return a logger with a default ColoredFormatter."""
    formatter = ColoredFormatter("%(asctime)s %(log_color)s%(levelname)-8s %(reset)s %(filename)s: %(message)s",
                                 datefmt='%Y-%m-%d %H:%M:%S',
                                 reset=True,
                                 log_colors={
                                     'DEBUG': 'blue',
                                     'INFO': 'green',
                                     'WARNING': 'yellow',
                                     'ERROR': 'red',
                                     'CRITICAL': 'red',
                                 })

    logger = logging.getLogger('project')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.info('logger init finished')
    return logger


def save_checkpoint(checkpoint_path, model, optimizer, epoch, logger):
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch}
    torch.save(state, checkpoint_path)
    logger.info('model saved to %s' % checkpoint_path)


def load_checkpoint(checkpoint_path, model, logger,device, optimizer=None):
    state = torch.load(checkpoint_path,map_location=device)
    model.load_state_dict(state['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    start_epoch = state['epoch']
    logger.info('model loaded from %s' % checkpoint_path)
    return start_epoch
