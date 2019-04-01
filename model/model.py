# -*- coding: utf-8 -*-
# @Time    : 2019/1/2 17:29
# @Author  : zhoujun
import torch
from torch import nn
import torch.nn.functional as F
from model.resnet import *

d = ['resnet50', 'resnet101', 'resnet152']
inplace = True


class PSENet(nn.Module):
    def __init__(self, backbone, result_num, scale: int = 1, pretrained=False):
        super(PSENet, self).__init__()
        assert backbone in d, 'backbone must in: {}'.format(d)
        self.scale = scale
        self.backbone = globals()[backbone](pretrained=pretrained)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        self.conv = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=inplace)
        )
        self.out_conv = nn.Conv2d(256, result_num, kernel_size=1, stride=1)

    def forward(self, input: torch.Tensor):
        _, _, H, W = input.size()
        c1, c2, c3, c4, c5 = self.backbone(input)
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        x = self._upsample_cat(p2, p3, p4, p5)
        x = self.conv(x)
        x = self.out_conv(x)

        x = F.interpolate(x, size=(H // self.scale, W // self.scale), mode='bilinear', align_corners=True)
        # x = torch.sigmoid(x)
        return x

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:], mode='bilinear', align_corners=True) + y

    def _upsample_cat(self, p2, p3, p4, p5):
        h, w = p2.size()[2:]
        p3 = F.interpolate(p3, size=(h, w), mode='bilinear', align_corners=True)
        p4 = F.interpolate(p4, size=(h, w), mode='bilinear', align_corners=True)
        p5 = F.interpolate(p5, size=(h, w), mode='bilinear', align_corners=True)
        return torch.cat([p2, p3, p4, p5], dim=1)


if __name__ == '__main__':
    device = torch.device('cuda:3')
    net = PSENet(backbone='resnet152', pretrained=False, result_num=8).to(device)
    x = torch.zeros(1, 3, 640, 640).to(device)
    y = net(x)
    print(y.shape)
