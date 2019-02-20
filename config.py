# -*- coding: utf-8 -*-
# @Time    : 2019/1/3 17:40
# @Author  : zhoujun

# data config
trainroot = '/data2/dataset/ICD15/train'
testroot = '/data2/dataset/ICD15/test'
output_dir = 'output/psenet_icd2015_gpu_resnet1521_lr0.0001_pse_crop_mask_normal'
data_shape = 640

# train config
gpu_id = '3'
workers = 6
start_epoch = 0
epochs = 600

train_batch_size = 4

lr = 1e-4
end_lr = 1e-7
lr_gamma = 0.1
lr_decay_step = [200, 400]
warmup_factor = 1.0 / 3
warmup_iters = 30
display_interval = 10
show_images_interval = 50
pretrained = True
restart_training = True
checkpoint = 'output/psenet_icd2015_gpu_resnet1521_lr0.0001_zj_resize_600_pred_new_eval_model_pse_crop/PSENet_415_loss0.284934_r0.316803_p0.586453_f10.411379.pth'

# net config
backbone = 'resnet152'
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
