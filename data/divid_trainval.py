'''
# Author: yangbinchao
# Date:   2021-12-10
# Email:  heroybc@qq.com
# Describe: 对指定数据集划分训练和验证，多进程，可保存图片目录或保存列表
'''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import cv2
import os
import argparse
import random
import json
from tqdm import tqdm
import numpy as np
from imutils import paths
from multiprocessing import Queue, Process
from shutil import copy

parser = argparse.ArgumentParser(description='Divide the training data set and the validation data set.')
parser.add_argument("--image", help='raw image path', default='./test', type=str)
parser.add_argument("--dir_save", help='save directory', default='./data_trainval', type=str)
parser.add_argument("--list_out", help='output list of train and val', default=False, type=bool)
parser.add_argument("--dir_out", help='output dir of train and val', default=True, type=bool)
parser.add_argument("--multi_process", help='use multi-progress', default=True, type=bool)
parser.add_argument("--rand_shuffle", help='use rand_shuffle', default=True, type=bool)
parser.add_argument("--num_worker", help='num_worker', default=4, type=int)
parser.add_argument("--rat", help='ratio of train', default=0.7, type=float)
args = parser.parse_args()

input_data = args.image
save_dir = args.dir_save
list_out = args.list_out
dir_out = args.dir_out
multi_process = args.multi_process
rand_shuffle = args.rand_shuffle
num_worker = args.num_worker
rat = args.rat

train_list = []  
val_list = [] 
train_dir = save_dir + "/train"
val_dir = save_dir + "/val"

def check_and_make_dir(dir_path):
    if os.path.exists(dir_path):
        print("{} already exists!!!".format(dir_path))
        #exit()
    else:
        os.makedirs(dir_path)
        assert os.path.exists(dir_path), dir_path

def split_task_for_workers(records, num_worker):
    """
    给每个worker分配要处理的数据
    """
    splits = {}
    for i_worker in range(num_worker):
        splits[i_worker] = []
    for i_data in range(len(records)):
        splits[i_data % num_worker].append(records[i_data])
    return splits

def handle(img_path, num_worker, img_dir, img_list):
    nr_records = len(img_path)
    pbar = tqdm(total=nr_records)   
    paral_queue = Queue()
    procs = []
    
    worker_records = split_task_for_workers(img_path, num_worker)

    for i_worker in range(num_worker):
        records = worker_records[i_worker]
        proc = Process(target=move,  # 目标函数
                       args=( records, img_dir, img_list),  # 目标函数参数
                       kwargs=dict(
                           #paral_queue=paral_queue,  # 字典参数
                       ))
        proc.start()
        procs.append(proc)
    for p in procs:
        p.join()


def move(img_paths, img_dir, img_list):  #i_worker
    for i in range(len(img_paths)):
        filename = img_paths[i]
        if dir_out:
            copy(filename, img_dir)
        if list_out:
            img_list.append(filename)

def load_data():
    check_and_make_dir(save_dir)  # 检查保存文件夹是否存在，存在则退出，不存在则建立
    check_and_make_dir(train_dir)
    check_and_make_dir(val_dir)
    
    print("load images...")
    img_paths = []
    img_paths += [el for el in paths.list_images(input_data)]  # 数据集图片路径，列出路径下的文件名或图片名并且存入list列表，进行for循环取出
    if rand_shuffle:
        print("shuffle image...")
        random.shuffle(img_paths)
    print('image data processing is kicked off...')
    print("%d images in total" % len(img_paths))
    return img_paths

def processing(img_path, img_dir, img_list):
    if multi_process:
        handle(img_path, num_worker, img_dir, img_list)
    else:
        move(img_path, img_dir, img_list)


if __name__ == '__main__' : 
    start = time.time()
    img_paths  = load_data()
    train_img_len = int(rat * len(img_paths))
    print("ratio of train and val: {},length of train: {}, length of val: {}".format(rat, train_img_len, len(img_paths)-train_img_len))
    if list_out is True:
        multi_process = False
        print("Note: if out file is list json, do not using multi-processing!")
    processing(img_paths[:train_img_len],train_dir,train_list)
    processing(img_paths[train_img_len:],val_dir,val_list)

    if list_out:
        with open(save_dir+'/train.json',  'a+',  encoding='utf-8') as f:  
                f.write(json.dumps(train_list,ensure_ascii=False,indent=1))
        with open(save_dir+'/val.json',  'a+',  encoding='utf-8') as f:  
                f.write(json.dumps(val_list,ensure_ascii=False,indent=1))
    end = time.time()
    print("processing time: {} s".format(end - start))
    print("\n>>> save dir: {}\n".format(save_dir))
