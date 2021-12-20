'''
# Author: yangbinchao
# Date:   2021-12-12
# Email:  heroybc@qq.com
# Describe: 损失函数ssim l1 soomth fsimc
'''


from __future__ import division
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from utils.inverse_warp import inverse_warp
from .fsim import FSIMc


def ssim(x, y):
    '''
    计算两张图像的结构相似度
    '''
    
    avepooling2d = torch.nn.AvgPool2d(3, stride=1, padding=[1, 1])
    mu_x = avepooling2d(x)
    mu_y = avepooling2d(y)
    # sigma_x = avepooling2d((x-mu_x)**2)
    # sigma_y = avepooling2d((y-mu_y)**2)
    # sigma_xy = avepooling2d((x-mu_x)*(y-mu_y))
    sigma_x = avepooling2d(x ** 2) - mu_x ** 2
    sigma_y = avepooling2d(y ** 2) - mu_y ** 2
    sigma_xy = avepooling2d(x * y) - mu_x * mu_y
    k1_square = 0.01 ** 2
    k2_square = 0.03 ** 2
    # L_square = 255**2
    L_square = 1
    SSIM_n = (2 * mu_x * mu_y + k1_square * L_square) * (2 * sigma_xy + k2_square * L_square)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + k1_square * L_square) * \
             (sigma_x + sigma_y + k2_square * L_square)
    SSIM = SSIM_n / SSIM_d
    return torch.clamp((1 - SSIM) / 2, 0, 1)

def l1(x, y):
    '''
    直接做差即可
    '''
    return x - y

def smooth(pred_map):
    '''
    计算单张图像的像素梯度，进行平滑损失计算，比如深度图平滑损失计算
    '''
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    if type(pred_map) not in [tuple, list]:
        pred_map = [pred_map]

    loss = 0
    weight = 1.

    for scaled_map in pred_map:
        dx, dy = gradient(scaled_map)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight
        weight /= 2.3  # don't ask me why it works better
    return loss

def fsimc(x, y):
    fsimc_loss = FSIMc()
    loss = fsimc_loss(x,y)
    return loss