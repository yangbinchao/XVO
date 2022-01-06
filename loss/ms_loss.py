'''
# Author: yangbinchao
# Date:   2021-12-12
# Email:  heroybc@qq.com
# Describe: 多尺度多权重的损失函数
'''

from __future__ import division
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils.inverse_warp import inverse_warp
from .ssim_l1_smooth import ssim, l1, fsimc, smooth


w_l1 = [0.1, 0.8, 1, 0.4, 0.1]
w_ssim = [0.1, 1.2, 1, 0.6, 0.1]
w_fsim = [0.1, 0.9, 1, 0.6, 0.1]
w_smooth = [1, 0.78, 0.25, 0.12, 0.06]


def one_scale_loss(tgt_img, ref_imgs, intrinsics, depth, pose, rotation_mode, padding_mode):
    '''
    针对单一尺度的损失计算
    '''
    assert(pose.size(1) == len(ref_imgs))
    # assert 的作用是现计算表达式 expression ，如果其值为假（即为0），那么它先向 stderr 打印一条出错信息,然后通过调用 abort 来终止程序运行。

    l1_loss = 0
    ssim_loss = 0
    fsim_loss = 0
    smooth_loss = 0
    b, _, h, w = depth.size()
    downscale = tgt_img.size(2)/h

    tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area')  # 通过插值改变图像大小与多尺度的深度图一致 
    ref_imgs_scaled = [F.interpolate(ref_img, (h, w), mode='area') for ref_img in ref_imgs]  # 多个参考图像使用 []
    intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)

    warped_imgs = []
    diff_maps = []

    for i, ref_img in enumerate(ref_imgs_scaled):
        current_pose = pose[:, i]
        ref_img_warped, valid_points = inverse_warp(ref_img, depth[:,0], current_pose,
                                                    intrinsics_scaled,
                                                    rotation_mode, padding_mode)
 
        x = tgt_img_scaled * valid_points.unsqueeze(1).float()
        y = ref_img_warped * valid_points.unsqueeze(1).float()
        
        ssim_loss += torch.mean(ssim(y, x))
        diff = l1(x,y)
        l1_loss += diff.abs().mean()
        # assert((l1_loss == l1_loss).item() == 1)
        # fsim_loss += torch.mean(fsimc(x,y))
        fsim_loss += diff.abs().mean()
        smooth_loss += smooth(depth)

        warped_imgs.append(ref_img_warped[0])
        diff_maps.append(diff[0])

    return l1_loss, ssim_loss, fsim_loss, smooth_loss, warped_imgs, diff_maps

def multi_scale_loss(tgt_img, ref_imgs, intrinsics, depth, pose, rotation_mode='euler', padding_mode='zeros'):
    '''
    设置多尺度的权重和计算损失
    '''
    total_l1_loss = 0
    total_ssim_loss = 0
    total_fsim_loss = 0
    total_smooth_loss = 0

    warped_results, diff_results = [], []
    if type(depth) not in [list, tuple]:
        depth = [depth]

    for i, d in enumerate(depth):
        l1_loss, ssim_loss, fsim_loss, smooth_loss, warped, diff = one_scale_loss(tgt_img, ref_imgs, intrinsics, d, pose, rotation_mode, padding_mode)

        total_l1_loss += w_l1[i] * l1_loss
        total_ssim_loss += w_ssim[i] * ssim_loss
        total_fsim_loss += w_fsim[i] * fsim_loss
        total_smooth_loss += w_smooth[i] * smooth_loss
        
        warped_results.append(warped)
        diff_results.append(diff)

    return total_l1_loss, total_ssim_loss, total_fsim_loss, total_smooth_loss, warped_results, diff_results