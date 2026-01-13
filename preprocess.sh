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
    --model_path $MODEL_PATH \
    --data_merge_path $DATA_MERGE_PATH \
    --preprocess_video_batch_size 4 \
    --seed 42 \
    --max_height 352 \
    --max_width 640 \
    --num_frames 77 \
    --dataloader_num_workers 0 \
    --output_dir=$OUTPUT_DIR \
    --samples_per_file 4 \
    --train_fps 25 \
    --flush_frequency 4 \
    --preprocess_task matrixgame
