'''
# Author: yangbinchao
# Date:   2021-12-15
# Email:  heroybc@qq.com
# Describe: 位姿估计评估，得出量化指标，可以输出评估结果
'''


import torch
from torch.autograd import Variable
from imageio import imread
from skimage.transform import resize
# from scipy.misc import imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
import time

from model import PoseExpNet, UNet3Plus_DeepSup_CGM, ShuffleNetV2
from utils.inverse_warp import pose_vec2mat

parser = argparse.ArgumentParser(description='Script for PoseNet testing with corresponding groundTruth from KITTI Odometry',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("pretrained_posenet", type=str, help="pretrained PoseNet path")
parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--min-depth", default=1e-3)
parser.add_argument("--max-depth", default=80)
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--sequences", default=['09'], type=str, nargs='*', help="sequences to test")
parser.add_argument("--output-dir", default=None, type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--rotation-mode", default='euler', choices=['euler', 'quat'], type=str)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

'''
python3 test_pose.py ./checkpoints/KITTI-sfm\,epoch_size1000/09-30-11\:38/exp_pose_model_best.pth.tar --dataset-dir /data/yangbinchaodata/KITTI-odometry/dataset/ --output-dir ./results/pose/ --sequences 09
'''
@torch.no_grad()
def main():
    args = parser.parse_args()
    from kitti_eval.pose_evaluation_utils import test_framework_KITTI as test_framework
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    weights = torch.load(args.pretrained_posenet)
    # seq_length = int(weights['state_dict']['conv1.0.weight'].size(1)/3)
    # print("注意！seq_length {}".format(seq_length))
    seq_length = 3
    pose_net = PoseExpNet(nb_ref_imgs=seq_length - 1).to(device)
    pose_net.load_state_dict(weights['state_dict'], strict=False)  # ybc , map_location={'cuda:1':'cuda:0'}

    dataset_dir = Path(args.dataset_dir)
    framework = test_framework(dataset_dir, args.sequences, seq_length)  # Zheli

    print('{} snippets to test'.format(len(framework)))
    errors = np.zeros((len(framework), 2), np.float32)  # 定义误差
    usetime_net = np.zeros((len(framework), 1), np.float64)
    usetime_trans = np.zeros((len(framework), 1), np.float64)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.makedirs_p()
        predictions_array = np.zeros((len(framework), seq_length, 3, 4))
        gtpose_array = np.zeros((len(framework), seq_length, 3, 4))  # ybc

    for j, sample in enumerate(tqdm(framework)):
        imgs = sample['imgs']

        h,w,_ = imgs[0].shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            imgs = [resize(img, (args.img_height, args.img_width)).astype(np.float32) for img in imgs]  # imresize

        imgs = [np.transpose(img, (2,0,1)) for img in imgs]

        ref_imgs = []
        for i, img in enumerate(imgs):
            img = torch.from_numpy(img).unsqueeze(0)
            img = ((img/255 - 0.5)/0.5).to(device)
            if i == len(imgs)//2:
                tgt_img = img
            else:
                ref_imgs.append(img)

        torch.cuda.synchronize()  # 开始计算时间
        start1 = time.time()

        poses = pose_net(tgt_img, ref_imgs)  # 输出位姿

        torch.cuda.synchronize()
        end1 = time.time()         # 结束计算时间

        usetime_net[j] = end1 - start1

        poses = poses.cpu()[0]
        poses = torch.cat([poses[:len(imgs)//2], torch.zeros(1,6).float(), poses[len(imgs)//2:]])  # 欧拉角和平移，3+3 一共6维

        inv_transform_matrices = pose_vec2mat(poses, rotation_mode=args.rotation_mode).numpy().astype(np.float64)
        '''
        输入： 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
        输出： A transformation matrix -- [B, 3, 4]
        '''

        rot_matrices = np.linalg.inv(inv_transform_matrices[:,:,:3])
        tr_vectors = -rot_matrices @ inv_transform_matrices[:,:,-1:]

        transform_matrices = np.concatenate([rot_matrices, tr_vectors], axis=-1)

        first_inv_transform = inv_transform_matrices[0]
        final_poses = first_inv_transform[:,:3] @ transform_matrices
        final_poses[:,:,-1:] += first_inv_transform[:,-1:]  # 得到位姿

        end2 = time.time()      # 记录计算位姿时间
        usetime_trans[j] = end2 - end1

        # print(final_poses[:,:,:3])
        if args.output_dir is not None:
            '''
            a9 = sample['poses']
            #print(a9.shape)
            a6 = np.sum(a9[:, :, :] * final_poses[:, :, :]) / np.sum(final_poses[:, :, :] ** 2)
            #print(final_poses.shape)
            pre_pose = final_poses[:, :, :]  # -1
            #print(final_poses.shape)
            pre_pose = a6 * pre_pose
            '''

            predictions_array[j] = final_poses
            gtpose_array[j] = sample['poses']  # ybc

        ATE, RE = compute_pose_error(sample['poses'], final_poses)
        errors[j] = ATE, RE

    mean_errors = errors.mean(0)
    std_errors = errors.std(0)
    error_names = ['ATE','RE']
    print('')
    print("Results")
    print("\t {:>10}, {:>10}".format(*error_names))
    print("mean \t {:10.4f}, {:10.4f}".format(*mean_errors))
    print("std \t {:10.4f}, {:10.4f}".format(*std_errors))

    usetime_net_mean = usetime_net.mean(0)
    usetime_net_std = usetime_net.std(0)
    usetime_trans_mean = usetime_trans.mean(0)
    print("位姿所用时间：")
    print("网络 mean \t {:10.4f}".format(*usetime_net_mean))
    print("网络 std \t {:10.4f}".format(*usetime_net_std))
    print("转换 mean \t {:10.4f}".format(*usetime_trans_mean))

    if args.output_dir is not None:
        np.save(output_dir/'predictions.npy', predictions_array)
        np.save(output_dir / 'gtpose.npy', gtpose_array)  # ybc


def compute_pose_error(gt, pred):
    RE = 0
    snippet_length = gt.shape[0]
    scale_factor = np.sum(gt[:,:,-1] * pred[:,:,-1])/np.sum(pred[:,:,-1] ** 2)
    ATE = np.linalg.norm((gt[:,:,-1] - scale_factor * pred[:,:,-1]).reshape(-1))
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose[:,:3] @ np.linalg.inv(pred_pose[:,:3])
        s = np.linalg.norm([R[0,1]-R[1,0],
                            R[1,2]-R[2,1],
                            R[0,2]-R[2,0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s,c)

    return ATE/snippet_length, RE/snippet_length


if __name__ == '__main__':
    main()
