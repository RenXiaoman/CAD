# !/bin/bash

# A100主机
export nnUNet_raw="/home/ikun_server/clib/PycharmProjects/CAD/dataset/PI-CAI/nnUNet_raw"
export nnUNet_preprocessed="/home/ikun_server/clib/PycharmProjects/CAD/dataset/PI-CAI/nnUNet_preprocessed"
export nnUNet_results="/home/ikun_server/clib/PycharmProjects/CAD/dataset/PI-CAI/nnUNet_results"
clear
# Docker
docker exec -it renxiaoman /bin/bash

cd /root/PycharmProjects/CAD/
clear
export nnUNet_raw="/root/PycharmProjects/CAD/dataset/PI-CAI/nnUNet_raw"
export nnUNet_preprocessed="/root/PycharmProjects/CAD/dataset/PI-CAI/nnUNet_preprocessed"
export nnUNet_results="/root/PycharmProjects/CAD/dataset/PI-CAI/nnUNet_results"
clear
# Check dataset integrity and preprocess
nnUNetv2_plan_and_preprocess -d 131 --verify_dataset_integrity


# Train
tmux
source ~/.bashrc
conda activate rxm
nnUNetv2_train 131 CONFIGURATION FOLD

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 131 3d_fullres 0 --npz
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 131 3d_fullres 1 --npz
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 131 3d_fullres 2 --npz
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 131 3d_fullres 3 --npz
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 131 3d_fullres 4 --npz

# Find the Best Configuration
nnUNetv2_find_best_configuration 131 -c CONFIGURATIONS
nnUNetv2_find_best_configuration 131 -c 3d_fullres 

# 单个
nnUNetv2_predict -d Dataset131_ProstatePI-CAI -i dataset/PI-CAI/nnUNet_raw/Dataset131_ProstatePI-CAI/imagesTs \
-o checkpoints/SegGland_nnUNet_PICAI/fold4 -f 4 -c 3d_fullres -p nnUNetPlans

# all集成
# nnUNetv2_predict -d Dataset131_ProstatePI-CAI -i dataset/PI-CAI/nnUNet_raw/Dataset131_ProstatePI-CAI/imagesTs \
# -o checkpoints/SegGland_nnUNet/all  -c 3d_fullres -f all