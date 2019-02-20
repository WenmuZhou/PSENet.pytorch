#encoding:utf-8
#Author:Snake
import cv2
import numpy as np
from skimage import exposure
class add_flip():
    def __init__(self,file_path):
        self.image = cv2.imread(file_path)
    # 左右镜像
    def _random_fliplr(self, random_fliplr=True):
        if random_fliplr and np.random.choice([True, True]):
            self.image = np.fliplr(self.image)  # 左右
        return self.image


    # 上下镜像
    def _random_flipud(self, random_flipud=True):
        if random_flipud and np.random.choice([True, True]):
            self.image = np.flipud(self.image)  # 上下
        return self.image

    # 改变光照
    def _random_exposure(self, random_exposure=True):
        if random_exposure and np.random.choice([True, False]):
            e_rate = np.random.uniform(0.3, 2)
            self.image = exposure.adjust_gamma(self.image, e_rate)
        return self.image

image_path = '/data/fxw/PSENet_data/test/img_2.jpg'

add_flip1 =add_flip(image_path)
print(add_flip1.image.shape)
image = add_flip1._random_exposure()
# image = add_flip1.image

cv2.namedWindow('fxw')
cv2.imshow('fxw',image)
cv2.waitKey()