#!/bin/bash

GPU_NUM=1 # 2,4,8
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
MODEL_TYPE="wan"
DATA_MERGE_PATH="mini_t2v_dataset/crush-smol_raw/merge.txt"
OUTPUT_DIR="mini_t2v_dataset/crush-smol_preprocessed"
VALIDATION_PATH="mini_t2v_dataset/crush-smol_raw/validation_t2v_64.json"

torchrun --nproc_per_node=$GPU_NUM \
    fastvideo/v1/pipelines/preprocess/v1_preprocess.py \
    --model_path $MODEL_PATH \
    --data_merge_path $DATA_MERGE_PATH \
    --preprocess_video_batch_size 8 \
    --max_height 768 \
    --max_width 1280 \
    --num_frames 77 \
    --dataloader_num_workers 0 \
    --output_dir=$OUTPUT_DIR \
    --model_type $MODEL_TYPE \
    --train_fps 16 \
    --validation_dataset_file $VALIDATION_PATH \
    --samples_per_file 8 \
    --flush_frequency 8 \
    --video_length_tolerance_range 5 \
    --preprocess_task "t2v" 