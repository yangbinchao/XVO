'''
# Author: yangbinchao
# Date:   2021-11-27
# Email:  heroybc@qq.com
# Describe: 可视化网络指定卷积层的权重，主要用于注意力机制的可视化
'''


import argparse
import cv2
import numpy as np
import torch
from torchvision import models

from pytorch_grad_cam import CAM, GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
from torchvision.models import resnet50

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_test_path', type=str, default='../../test_img/cbam_test_img_example/test.jpg',help='Input image path')
    parser.add_argument('--method', type=str, default='gradcam', help='Can be gradcam/gradcam++/scorecam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using gpu cuda for acceleration')
    else:
        print('Using cpu for computation')

    return args

if __name__ == '__main__':
    args = get_args()

    model = resnet50(pretrained=True)
    target_layer = model.layer4[-1]
    print(model.layer4[-1])

    target_layer = model.layer4[-1]

    cam = CAM(model=model, 
              target_layer=target_layer,
              use_cuda=args.use_cuda)

    # Create an input tensor image for your model
    rgb_img = cv2.imread(args.image_test_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None
    grayscale_cam = cam(input_tensor=input_tensor, 
                        method=args.method,
                        target_category=target_category)


    cam_image = show_cam_on_image(rgb_img, grayscale_cam)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    gb = gb_model(input_tensor, target_category=target_category)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    cv2.imwrite(f'./demo/{args.method}_cam.jpg', cam_image)
    cv2.imwrite(f'./demo/{args.method}_gb.jpg', gb)
    cv2.imwrite(f'./demo/{args.method}_cam_gb.jpg', cam_gb)