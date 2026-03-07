#!/bin/bash

GPU_NUM=1 # 2,4,8
MODEL_PATH="weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers"
DATA_MERGE_PATH="mc_wasd_10/merge.txt"
OUTPUT_DIR="mc_wasd_10/preprocessed/"

# export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export RANK=0
export WORLD_SIZE=1

python fastvideo/pipelines/preprocess/v1_preprocess.py \
    --model_path $MODEL_PATH \
    --data_merge_path $DATA_MERGE_PATH \
    --preprocess_video_batch_size 10 \
    --seed 42 \
    --max_height 352 \
    --max_width 640 \
    --num_frames 77 \
    --dataloader_num_workers 0 \
    --output_dir=$OUTPUT_DIR \
    --samples_per_file 10 \
    --train_fps 25 \
    --flush_frequency 10 \
    --preprocess_task wangame