# This project is no longer supported

# Note:
1. base on python 3.8 and pytorch.
2. all coding should care your file path.
3. normally, the time-using is 2 days and 2 Titan X GPU are used, if the train time overtime, please to check the img_size and the number of datasets.

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
### depth inference
> bash bash/run_depth.sh
or
> python run_depth.py --output-disp --pretrained ./checkpoints/KITTI-sfm\,epoch_size1000/09-22-09\:32/dispnet_model_best.pth.tar --dataset-dir ./depth_eva_image/sem_test/ --output-dir ./depth_eva_image/sem_test/

### depth eval
> bash bash/test_depth.sh
or
> python test_disp.py --pretrained-dispnet ckpts/KITTI-sfm/12-13-20:30/dispnet_model_best.pth.tar --pretrained-posenet ckpts/KITTI-sfm/12-13-20:30/exp_pose_model_best.pth.tar --dataset-dir /data/yangbinchao_data/KITTI-raw/rawdata/ --dataset-list ./kitti_eval/test_files_eigen.txt
### pose eval
> bash bash/test_pose.sh
or
> python test_pose.py ckpts/KITTI-sfm/12-13-20:30/exp_pose_model_best.pth.tar --dataset-dir /data/yangbinchao_data/KITTI-odometry/dataset/ --output-dir test_result/pose/ --sequences 09

visual pose result
> bash bash/evo_traj.sh
or
> evo_traj kitti test_result/pose//kitti_odometry_test/09.txt --ref=test_result/pose//kitti_odometry_gt/09.txt -p --plot_mode=xz -as
> evo_rpe kitti 09.txt 09.txt -p --plot_mode=xyz -vas -r=full
> evo_ape kitti 09.txt 09.txt -p --plot_mode=xyz -vas -r=full

## Result
- null

# Reference
- https://github.com/avBuffer/UNet3plus_pth 
- https://github.com/ClementPinard/SfmLearner-Pytorch
- https://github.com/mikhailiuk/pytorch-fsim

