#!/bin/bash

GPU_NUM=1 # 2,4,8
MODEL_PATH="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
MODEL_TYPE="wan"
DATA_MERGE_PATH="data/crush-smol/merge.txt"
OUTPUT_DIR="data/crush-smol_processed_i2v/"
VALIDATION_PATH="examples/training/finetune/wan_i2v_14b_480p/crush_smol/validation.json"

torchrun --nproc_per_node=$GPU_NUM \
    fastvideo/v1/pipelines/preprocess/v1_preprocess.py \
    --model_path $MODEL_PATH \
    --data_merge_path $DATA_MERGE_PATH \
    --preprocess_video_batch_size 8 \
    --max_height 480 \
    --max_width 832 \
    --num_frames 77 \
    --dataloader_num_workers 0 \
    --output_dir=$OUTPUT_DIR \
    --model_type $MODEL_TYPE \
    --train_fps 16 \
    --validation_dataset_file $VALIDATION_PATH \
    --samples_per_file 8 \
    --flush_frequency 8 \
    --video_length_tolerance_range 5 \
    --preprocess_task "i2v" 