'''
# Author: yangbinchao
# Date:   2021-12-10
# Email:  heroybc@qq.com
# Describe: 对指定数据集进行筛选、分类
'''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numpy.lib.type_check import imag
from imutils import paths
import numpy as np
import cv2
import os
import argparse
import random
from tqdm import tqdm
from shutil import copy
from multiprocessing import Queue, Process


parser = argparse.ArgumentParser(description='fliter the data set under some rule.')
parser.add_argument("--image", help='raw image path', default='./test', type=str)
parser.add_argument("--multi_process", help='use multi-progress', default=True, type=bool)
parser.add_argument("--rand_shuffle", help='use rand_shuffle', default=True, type=bool)
parser.add_argument("--num_worker", help='num_worker', default=4, type=int)
parser.add_argument("--rat", help='ratio of train', default=0.7, type=float)
args = parser.parse_args()

input_data = args.image
multi_process = args.multi_process
rand_shuffle = args.rand_shuffle
num_worker = args.num_worker
rat = args.rat

def filter_name(filename):
    '''
    按照命名规则进行划分
    ''' 
    label_dir = '/root/yangbinchao/program/data/APM/test/label'
    img_dir = '/root/yangbinchao/program/data/APM/test/image'
    basename = os.path.basename(filename)
    imgname, suffix = os.path.splitext(basename)
    label = imgname.split("_")[-1]
    if label == "matte":
        copy(filename, label_dir)
    else:
        copy(filename, img_dir)

def load_data():
    print("load images...")
    img_paths = []
    img_paths += [el for el in paths.list_images(input_data)]  
    print('image data processing is kicked off...')
    print("%d images in total" % len(img_paths))
    return img_paths

if __name__ == '__main__' :
    img_paths = load_data()
    for i in tqdm(range(len(img_paths))):
        filename = img_paths[i]
        filter_name(filename)

