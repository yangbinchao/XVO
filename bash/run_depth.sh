python run_depth.py \
    --output-disp \
    --pretrained ./ckpts/KITTI-sfm/12-13-20:30/dispnet_model_best.pth.tar \
    --dataset-dir /data/yangbinchao_data/KITTI-raw/rawdata/ \
    --dataset-list kitti_eval/test_files_eigen.txt \
    --output-dir test_result/depth/kitti_eigen_test/12-13-20:30/
