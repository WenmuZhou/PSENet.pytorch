# -*- coding: utf-8 -*-
# @Time    : 2019/1/3 16:19
# @Author  : zhoujun
import torch
from torch import nn


class PSELoss(nn.Module):
    def __init__(self, Lambda, ratio=3, reduction='mean'):
        """Implement PSE Loss.
        """
        super(PSELoss, self).__init__()
        assert reduction in ['mean', 'sum'], " reduction must in ['mean','sum']"
        self.Lambda = Lambda
        self.ratio = ratio
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, labels, training_mask):
        """
        loss 由两部分组成 L_c ,L_s
        L_c 表示对完整文本实例的loss，会根据OHEM算法选择背景类
        L_s 是针对缩放文本实例的loss
        :param preds: 预测结果
        :param labels: ground truth
        :return: loss 值
        """
        bs, n, w, h = preds.size()
        preds = preds.contiguous().view(bs, n, w * h)  # b,c,h,w -> b,c,h*w
        labels = labels.contiguous().view(bs, n, w * h)
        training_mask = training_mask.contiguous().view(bs, w * h).to(torch.float)

        all_loss_c = []
        all_loss_s = []
        all_loss = []
        for pred, label, mask in zip(preds, labels, training_mask):
            L_c, L_s, loss = self.single_sample_loss(pred, label, mask)
            all_loss_c.append(L_c)
            all_loss_s.append(L_s)
            all_loss.append(loss)
        all_loss_c = torch.stack(all_loss_c)
        all_loss_s = torch.stack(all_loss_s)
        all_loss = torch.stack(all_loss)
        if self.reduction == 'mean':
            all_loss_c = all_loss_c.mean()
            all_loss_s = all_loss_s.mean()
            all_loss = all_loss.mean()
        elif self.reduction == 'sum':
            all_loss_c = all_loss_c.sum()
            all_loss_s = all_loss_s.sum()
            all_loss = all_loss.sum()
        return all_loss_c, all_loss_s, all_loss

    def single_sample_loss(self, pred, label, training_mask):
        pred = torch.sigmoid(pred)
        pred_n = pred[-1] * training_mask
        label_n = label[-1] * training_mask
        M = self.cal_M(pred_n)
        L_c = 1 - self.cal_D(pred_n.unsqueeze(0) * M, label_n.unsqueeze(0) * M)
        # 计算L_s
        W = (pred_n >= 0.5).float()
        L_s = 1 - self.cal_D(pred[:-1] * W, label[:-1] * W)
        all_loss = self.Lambda * L_c + (1 - self.Lambda) * L_s
        return L_c, L_s, all_loss

    def cal_D(self, S, G):
        """
        计算每个样本的D
        :param S:
        :param G:
        :return:
        """
        D = []
        for s, g in zip(S, G):
            D.append(2 * torch.sum(s * g) / (torch.sum(s * s) + torch.sum(g * g) + 1e-6))
        if not D:
            return torch.tensor(1).float().cuda()
        return torch.mean(torch.stack(D))

    def cal_M(self, pred_n):
        """
        使用OHEM算法选择参与计算loss的文本和背景像素矩阵,单个样本
        :param pred_n: 完整文本实例的预测结果
        :param label_n: 完整文本实例的ground truth
        :return:
        """
        # 计算选择背景区域像素点个数
        pos_mask = (pred_n >= 0.5)
        neg_mask = (pred_n < 0.5)
        n_pos = pos_mask.sum().int().item()
        n_neg = neg_mask.sum().int().item()
        if n_neg > n_pos * self.ratio:
            n_neg = n_pos * self.ratio
            # 从预测图里拿到背景像素的分数
            zero_predict_score = pred_n.masked_select(neg_mask)
            # 按照OHEM的比例选取背景像素的x,y索引
            value, _ = zero_predict_score.topk(n_neg)
            if len(value):
                threshold = value[-1]
                M = pred_n >= threshold
            else:
                # 当n_pos==0时
                M = torch.ones_like(pred_n).to(pred_n.device)
        else:
            M = torch.ones_like(pred_n).to(pred_n.device)
        return M.float()
