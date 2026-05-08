# !/bin/bash

# A100主机
export nnUNet_raw="/home/ikun_server/clib/PycharmProjects/CAD/dataset/AHCDU/二分类图像/nnUNet_raw"
export nnUNet_preprocessed="/home/ikun_server/clib/PycharmProjects/CAD/dataset/AHCDU/二分类图像/nnUNet_preprocessed"
export nnUNet_results="/home/ikun_server/clib/PycharmProjects/CAD/dataset/AHCDU/二分类图像/nnUNet_results"

# Docker
docker exec -it renxiaoman /bin/bash

cd /root/PycharmProjects/CAD/
clear
export nnUNet_raw="/root/PycharmProjects/CAD/dataset/AHCDU/二分类图像/nnUNet_raw"
export nnUNet_preprocessed="/root/PycharmProjects/CAD/dataset/AHCDU/二分类图像/nnUNet_preprocessed"
export nnUNet_results="/root/PycharmProjects/CAD/dataset/AHCDU/二分类图像/nnUNet_results"

# Check dataset integrity and preprocess
nnUNetv2_plan_and_preprocess -d 130 --verify_dataset_integrity



# Train
tmux
source ~/.bashrc
conda activate rxm
nnUNetv2_train 130 CONFIGURATION FOLD

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 130 3d_fullres 0 --npz
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 130 3d_fullres 1 --npz
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 130 3d_fullres 2 --npz
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 130 3d_fullres 3 --npz
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 130 3d_fullres 4 --npz

# Find the Best Configuration
nnUNetv2_find_best_configuration 130 -c CONFIGURATIONS
nnUNetv2_find_best_configuration 130 -c 3d_fullres 

# 单个
nnUNetv2_predict -d Dataset130_ProstateAHCDU -i dataset/AHCDU/二分类图像/nnUNet_raw/Dataset130_ProstateAHCDU/imagesTs \
-o checkpoints/SegGland_nnUNet/fold4 -f 4 -c 3d_fullres -p nnUNetPlans

# all集成
nnUNetv2_predict -d Dataset130_ProstateAHCDU -i dataset/AHCDU/二分类图像/nnUNet_raw/Dataset130_ProstateAHCDU/imagesTs \
-o checkpoints/SegGland_nnUNet/all  -c 3d_fullres -f all