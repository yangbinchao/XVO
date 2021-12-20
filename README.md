# My thesis in here!
Welcome to my thesis code and other things from tyut.
I hope I could done this with gooooood job!

# Note:
1. this code is yangbinchao's thesis code. 
2. the record in there: https://note.youdao.com/s/YoEzDG9F
3. base on python 3.8 and pytorch.
4. 
5. 

# Use:
Notice: all coding should care your file path. 

## Installation
> pip3 install -r requirements.txt

## Utils 
1. visual attention mechanism through grad cam.
> python ./utils/grad_cam_vis.py --image_test_path ./path

2. compare the structural similarity of two image.
> python ./utils/ssim_img.py 

## Train
> python train.py ../../program/data/KITTI-sfm --log-output --with-gt --with-pose

## Test
### depth eval
> python test_disp.py --pretrained-dispnet ckpts/KITTI-sfm/12-13-20:30/dispnet_model_best.pth.tar --pretrained-posenet ckpts/KITTI-sfm/12-13-20:30/exp_pose_model_best.pth.tar --dataset-dir /data/yangbinchao_data/KITTI-raw/rawdata/ --dataset-list ./kitti_eval/test_files_eigen.txt
### pose eval
> python test_pose.py ckpts/KITTI-sfm/12-13-20:30/exp_pose_model_best.pth.tar --dataset-dir /data/yangbinchao_data/KITTI-odometry/dataset/ --output-dir test_result/pose/ --sequences 09
'''

## Result

## Nvidia xavier agx
1. env: conda activate torch18
2. workpath: ~/yangbinchao
3. 

# TYUT
1. this project is running in the server 1 of the ISC-lab, the envs is "torch", you should use order is "source actiavte torch" to open torch envs.
2. normally, the time-using is 2 days and 2 GPU are used, if the train time overtime, please to check the img_size and the number of datasets.
3. 

# Reference
- Thanks for https://github.com/avBuffer/UNet3plus_pth 
- Thanks for https://github.com/ClementPinard/SfmLearner-Pytorch
- Thanks for https://github.com/mikhailiuk/pytorch-fsim

