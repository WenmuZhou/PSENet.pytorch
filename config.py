# -*- coding: utf-8 -*-
# @Time    : 2019/1/3 17:40
# @Author  : zhoujun

# data config
trainroot = '/data2/dataset/ICD15/train'
testroot = '/data2/dataset/ICD15/test'
output_dir = 'output/psenet_icd2015_resnet50_my_loss_0.001_author_crop_adam_newcrop_authorloss'
data_shape = 640

# train config
gpu_id = '3'
workers = 6
start_epoch = 0
epochs = 600

train_batch_size = 8

lr = 1e-3
end_lr = 1e-7
lr_gamma = 0.1
lr_decay_step = [200,400]
weight_decay = 5e-4
warm_up_epoch = 6
warm_up_lr = lr * lr_gamma

display_interval = 10
show_images_interval = 50
pretrained = True
restart_training = True
checkpoint = 'output/psenet_icd2015_resnet152_my_loss_0.0001_author_crop/PSENet_446_loss0.257294_r0.715936_p0.713874_f10.714904.pth'

# net config
backbone = 'resnet50'
Lambda = 0.7
n = 6
m = 0.5
OHEM_ratio = 3

# random seed
seed = 2


def print():
    from pprint import pformat
    tem_d = {}
    for k, v in globals().items():
        if not k.startswith('_') and not callable(v):
            tem_d[k] = v
    return pformat(tem_d)
