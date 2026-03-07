#!/bin/bash

export PYTHONPATH="/mnt/fast-disks/hao_lab/kaiqin/FastVideo_wangame:$PYTHONPATH"

GPU_NUM=1 # 2,4,8
MODEL_PATH="./Wan2.1-Fun-1.3B-InP-Diffusers"
DATA_MERGE_PATH="../traindata_0209_1500/ode_init_mc/merge.txt"
OUTPUT_DIR="../traindata_0209_1500/ode_init_mc/preprocessed"

torchrun --nproc_per_node=$GPU_NUM \
    fastvideo/pipelines/preprocess/v1_preprocess.py \
    --model_path $MODEL_PATH \
    --data_merge_path $DATA_MERGE_PATH \
    --preprocess_video_batch_size 1 \
    --seed 42 \
    --max_height 352 \
    --max_width 640 \
    --num_frames 81 \
    --dataloader_num_workers 0 \
    --output_dir=$OUTPUT_DIR \
    --samples_per_file 8 \
    --train_fps 25 \
    --flush_frequency 8 \
    --preprocess_task wangame_ode_trajectory &
