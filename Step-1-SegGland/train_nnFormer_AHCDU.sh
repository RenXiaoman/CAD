#!/bin/bash
conda activate rxm

export nnFormer_raw_data_base="dataset/AHCDU/二分类图像/nnUNet_raw"
export nnFormer_preprocessed="dataset/AHCDU/二分类图像//nnUNet_preprocessed"
export RESULTS_FOLDER="checkpoints/SegGland_nnFormer_AHCDU"
clear

export nnFormer_raw_data_base="/root/PycharmProjects/nnFormer/DATASET/nnFormer_raw"
export nnFormer_preprocessed="/root/PycharmProjects/nnFormer/DATASET/nnFormer_preprocessed"
export RESULTS_FOLDER="/root/PycharmProjects/nnFormer/DATASET/nnFormer_trained_models"
clear
nnFormer_plan_and_preprocess -t 13


while getopts 'c:n:t:r:p' OPT; do
    case $OPT in
        c) cuda=$OPTARG;;
        n) name=$OPTARG;;
        f) f=$OPTARG;;
        r) train="true";;
        p) predict="true";;
        

    esac
done
echo $name	


# nnFormer_find_best_configuration -m 3d_fullres -t 4 -tr nnFormerTrainerV2

if ${train}
then
	unset LD_LIBRARY_PATH
	cd /home/ikun_server/clib/PycharmProjects/nnFormer/

    # AHCDU 
	CUDA_VISIBLE_DEVICES=1 nnFormer_train 3d_fullres nnFormerTrainerV2 13 0
    CUDA_VISIBLE_DEVICES=1 nnFormer_train 3d_fullres nnFormerTrainerV2 13 1
    CUDA_VISIBLE_DEVICES=2 nnFormer_train 3d_fullres nnFormerTrainerV2 13 2
    CUDA_VISIBLE_DEVICES=3 nnFormer_train 3d_fullres nnFormerTrainerV2 13 3
    CUDA_VISIBLE_DEVICES=0 nnFormer_train 3d_fullres nnFormerTrainerV2 13 4

    # PICAI 
	CUDA_VISIBLE_DEVICES=1 nnFormer_train 3d_fullres nnFormerTrainerV2 131 0
    CUDA_VISIBLE_DEVICES=1 nnFormer_train 3d_fullres nnFormerTrainerV2 131 1
    CUDA_VISIBLE_DEVICES=2 nnFormer_train 3d_fullres nnFormerTrainerV2 131 2
    CUDA_VISIBLE_DEVICES=3 nnFormer_train 3d_fullres nnFormerTrainerV2 131 3
    CUDA_VISIBLE_DEVICES=0 nnFormer_train 3d_fullres nnFormerTrainerV2 131 4


fi

if ${predict}
then


	cd /home/ikun_server/clib/PycharmProjects/nnFormer/DATASET/nnFormer_raw/nnFormer_raw_data/Task004_prostate
	CUDA_VISIBLE_DEVICES=2 nnFormer_predict -i imagesTs -o inferTs/prostat_4 -m 3d_fullres -t 4 -f 4 -chk model_best -tr nnFormerTrainerV2
	# python inference_acdc.py ${name}
fi

# bash train_inference.sh -c 0  -t 4

