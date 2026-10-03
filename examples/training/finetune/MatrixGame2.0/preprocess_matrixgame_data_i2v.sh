#!/bin/bash

GPU_NUM=1 # 2,4,8
MODEL_PATH="Matrix-Game-2.0-Base-Diffusers"
DATA_MERGE_PATH="footsies-dataset/merge.txt"
OUTPUT_DIR="footsies-dataset/preprocessed/"

# export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export RANK=0
export WORLD_SIZE=1

python fastvideo/pipelines/preprocess/v1_preprocess.py \
    --config examples/training/finetune/MatrixGame2.0/preprocess_matrixgame_data_i2v.yaml \
    --model_path $MODEL_PATH \
    --preprocess.data_merge_path $DATA_MERGE_PATH \
    --preprocess.dataset_output_dir=$OUTPUT_DIR
