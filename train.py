# -*- coding: utf-8 -*-
# @Time    : 2018/6/11 15:54
# @Author  : zhoujun
import cv2
import os
import config

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

import shutil
import time
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
import torch.utils.data as Data
from torchvision import transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from dataset.data_utils import MyDataset
from model import PSENet
from model.loss import PSELoss
# from PSELoss import PSELoss
from utils.utils import load_checkpoint, save_checkpoint, setup_logger
from utils.lr_scheduler import WarmupMultiStepLR
from predict import decode
from cal_recall import cal_recall_precison_f1


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def train_epoch(net, optimizer, scheduler, train_loader, device, criterion, epoch, all_step, writer, logger):
    net.train()
    train_loss = 0.
    start = time.time()
    scheduler.step()
    for i, (images, labels, training_mask) in enumerate(train_loader):
        cur_batch = images.size()[0]
        images, labels, training_mask = images.to(device), labels.to(device), training_mask.to(device)
        # Forward
        y1 = net(images)
        loss_c, loss_s, loss = criterion(y1, labels, training_mask)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        loss_c = loss_c.item()
        loss_s = loss_s.item()
        loss = loss.item()
        cur_step = epoch * all_step + i
        writer.add_scalar(tag='Train/loss_c', scalar_value=loss_c, global_step=cur_step)
        writer.add_scalar(tag='Train/loss_s', scalar_value=loss_s, global_step=cur_step)
        writer.add_scalar(tag='Train/loss', scalar_value=loss, global_step=cur_step)
        writer.add_scalar(tag='Train/lr', scalar_value=scheduler.get_lr()[0], global_step=cur_step)

        if i % config.display_interval == 0:
            batch_time = time.time() - start
            logger.info(
                '[{}/{}], [{}/{}], batch_loss_c: {:.4f}, batch_loss_s: {:.4f}, batch_loss: {:.4f}, time:{:.4f}, lr:{}'.format(
                    epoch, config.epochs, i, all_step, loss_c, loss_s, loss, batch_time,
                    str(scheduler.get_lr()[0])))
            start = time.time()

        if i % config.show_images_interval == 0:
            # show images on tensorboard
            x = vutils.make_grid(images.detach().cpu(), nrow=4, normalize=True, scale_each=True, padding=20)
            writer.add_image(tag='input/image', img_tensor=x, global_step=cur_step)

            show_label = labels.detach().cpu()
            b, c, h, w = show_label.size()
            show_label = show_label.reshape(b * c, h, w)
            show_label = vutils.make_grid(show_label.unsqueeze(1), nrow=config.n, normalize=False, padding=20,
                                          pad_value=1)
            writer.add_image(tag='input/label', img_tensor=show_label, global_step=cur_step)

            show_y = y1.detach().cpu()
            b, c, h, w = show_y.size()
            show_y = show_y.reshape(b * c, h, w)
            show_y = vutils.make_grid(show_y.unsqueeze(1), nrow=config.n, normalize=False, padding=20, pad_value=1)
            writer.add_image(tag='output/preds', img_tensor=show_y, global_step=cur_step)
    return train_loss


def eval(model, save_path, test_path, device):
    model.eval()
    # torch.cuda.empty_cache()  # speed up evaluating after training finished
    img_path = os.path.join(test_path, 'img')
    gt_path = os.path.join(test_path, 'gt')
    if os.path.exists(save_path):
        shutil.rmtree(save_path, ignore_errors=True)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 预测所有测试图片
    img_paths = [os.path.join(img_path, x) for x in os.listdir(img_path)]
    pbar = tqdm(total=len(img_paths), desc='test model:')
    for img_path in img_paths:
        img_name = os.path.basename(img_path).split('.')[0]
        save_name = os.path.join(save_path, 'res_' + img_name + '.txt')

        assert os.path.exists(img_path), 'file is not exists'
        img = cv2.imread(img_path)
        # 将图片由(w,h)变为(1,img_channel,h,w)
        tensor = transforms.ToTensor()(img)
        tensor = tensor.unsqueeze_(0)
        tensor = tensor.to(device)
        with torch.no_grad():
            preds = model(tensor)
            _, boxes_list = decode(preds, num_pred=-1)
        boxes_list = boxes_list.astype(np.int)[0]
        np.savetxt(save_name, boxes_list.reshape(-1, 8), delimiter=',', fmt='%d')
        pbar.update(1)
    pbar.close()
    # 开始计算 recall precision f1
    result_dict = cal_recall_precison_f1(gt_path, save_path)
    return result_dict['recall'], result_dict['precision'], result_dict['hmean']


def main():
    if config.output_dir is None:
        config.output_dir = 'output'
    if config.restart_training:
        shutil.rmtree(config.output_dir, ignore_errors=True)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    logger = setup_logger(os.path.join(config.output_dir, 'train_log'))
    logger.info(config.print())

    torch.manual_seed(config.seed)  # 为CPU设置随机种子
    if config.gpu_id is not None and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info('train with gpu {} and pytorch {}'.format(config.gpu_id, torch.__version__))
        device = torch.device("cuda:0")
        torch.cuda.manual_seed(config.seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(config.seed)  # 为所有GPU设置随机种子
    else:
        logger.info('train with cpu and pytorch {}'.format(torch.__version__))
        device = torch.device("cpu")

    train_data = MyDataset(config.trainroot, data_shape=config.data_shape, n=config.n, m=config.m,
                           transform=transforms.ToTensor())
    train_loader = Data.DataLoader(dataset=train_data, batch_size=config.train_batch_size, shuffle=True,
                                   num_workers=int(config.workers))

    writer = SummaryWriter(config.output_dir)
    model = PSENet(backbone=config.backbone, pretrained=config.pretrained, result_num=config.n)
    if not config.pretrained and not config.restart_training:
        model.apply(weights_init)

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    # dummy_input = torch.autograd.Variable(torch.Tensor(1, 3, 600, 800).to(device))
    # writer.add_graph(model=model, input_to_model=dummy_input)
    criterion = PSELoss(Lambda=config.Lambda, ratio=config.OHEM_ratio, reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    if config.checkpoint != '' and not config.restart_training:
        start_epoch = load_checkpoint(config.checkpoint, model, logger, device, optimizer)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.lr_decay_step, gamma=config.lr_gamma,
                                                         last_epoch=start_epoch)
    else:
        start_epoch = config.start_epoch
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.lr_decay_step, gamma=config.lr_gamma)

    # warmup for multi gpu
    # milestones = [step // num_gpus for step in config.lr_decay_step]
    # scheduler = WarmupMultiStepLR(optimizer=optimizer, milestones=milestones, gamma=config.lr_gamma,
    #                               warmup_factor=config.warmup_factor, warmup_iters=config.warmup_iters)

    all_step = len(train_loader)
    logger.info('train dataset has {} samples,{} in dataloader'.format(train_data.__len__(), all_step))
    epoch = 0
    best_model = {'recall': 0, 'precision': 0, 'f1': 0, 'model': ''}
    try:
        for epoch in range(start_epoch + 1, config.epochs):
            start = time.time()
            train_loss = train_epoch(model, optimizer, scheduler, train_loader, device, criterion, epoch, all_step,
                                     writer, logger)
            logger.info('[{}/{}], train_loss: {:.4f}, time: {:.4f}, lr: {}'.format(
                epoch, config.epochs, train_loss / all_step, time.time() - start, str(scheduler.get_lr()[0])))

            recall, precision, f1 = eval(model, os.path.join(config.output_dir, 'output'), config.testroot, device)
            logger.info('test: recall: {:.6f}, precision: {:.6f}, f1: {:.6f}'.format(recall, precision, f1))

            net_save_path = '{}/PSENet_{}_loss{:.6f}_r{:.6f}_p{:.6f}_f1{:.6f}.pth'.format(config.output_dir, epoch,
                                                                                          train_loss / all_step, recall,
                                                                                          precision,
                                                                                          f1)
            save_checkpoint(net_save_path, model, optimizer, epoch, logger)
            if f1 > best_model['f1']:
                best_model['recall'] = recall
                best_model['precision'] = precision
                best_model['f1'] = f1
                best_model['model'] = net_save_path
            writer.add_scalar(tag='Test/recall', scalar_value=recall, global_step=epoch)
            writer.add_scalar(tag='Test/precision', scalar_value=precision, global_step=epoch)
            writer.add_scalar(tag='Test/f1', scalar_value=f1, global_step=epoch)
        writer.close()
    except KeyboardInterrupt:
        save_checkpoint('{}/final.pth'.format(config.output_dir), model, optimizer, epoch, logger)
    finally:
        if best_model['model']:
            shutil.copy(best_model['model'],
                        '{}/best_r{:.6f}_p{:.6f}_f1{:.6f}.pth'.format(config.output_dir, best_model['recall'],
                                                                      best_model['precision'], best_model['f1']))
            logger.info(best_model)


if __name__ == '__main__':
    main()
