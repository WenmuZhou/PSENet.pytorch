# -*- coding: utf-8 -*-
# @Time    : 2018/6/11 15:54
# @Author  : zhoujun
import torch
import shutil
import numpy as np
import config
import os
from tqdm import tqdm
from model import PSENet
from predict import Pytorch_model
from cal_recall.script import cal_recall_precison_f1

torch.backends.cudnn.benchmark = True


def main(model_path, path, save_path, gpu_id):
    if os.path.exists(save_path):
        shutil.rmtree(save_path, ignore_errors=True)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img_paths = [os.path.join(path, x) for x in os.listdir(path)]
    net = PSENet(backbone='resnet152', pretrained=False, result_num=config.n)
    model = Pytorch_model(model_path, net=net, scale=1, gpu_id=gpu_id)
    pbar = tqdm(total=len(img_paths))
    for img_path in img_paths:
        img_name = os.path.basename(img_path).split('.')[0]
        save_name = os.path.join(save_path, 'res_' + img_name + '.txt')
        _, preds = model.predict(img_path)
        np.savetxt(save_name, preds.reshape(-1, 8), delimiter=',', fmt='%d')
        pbar.update(1)
    pbar.close()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str('2')
    model_path= 'output/psenet_icd2015_gpu_resnet1521_lr0.0001_pse_crop_mask/PSENet_356_loss0.259558_r0.378430_p0.577093_f10.457110.pth'
    data_path = '/data2/dataset/ICD15/test/img'
    gt_path = '/data2/dataset/ICD15/test/gt'
    save_path = './result2'
    gpu_id = 0
    main(model_path, data_path, save_path, gpu_id=gpu_id)
    result = cal_recall_precison_f1(gt_path=gt_path, result_path=save_path)
    print(result)
