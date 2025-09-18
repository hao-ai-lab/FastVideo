#!/bin/bash

GPU_NUM=1 # 2,4,8
# MODEL_PATH="Wan-AI/Wan2.2-T2V-A14B-Diffusers"
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
MODEL_TYPE="wan"
DATA_MERGE_PATH="examples/dataset/crush_smol/crush_smol_prompts.txt"
DATA_MERGE_PATH="test.txt"
DATA_MERGE_PATH="/mnt/weka/home/hao.zhang/wl/Self-Forcing/prompts/vidprom_1.txt"
OUTPUT_DIR="data/crush-smol_processed_t2v_a14b_ode_init/"

torchrun --nproc_per_node=$GPU_NUM \
    fastvideo/pipelines/preprocess/v1_preprocess.py \
    --model_path $MODEL_PATH \
    --data_merge_path $DATA_MERGE_PATH \
    --preprocess_video_batch_size 1 \
    --seed 42 \
    --max_height 480 \
    --max_width 832 \
    --num_frames 81 \
    --flow_shift 12.0 \
    --dataloader_num_workers 0 \
    --output_dir=$OUTPUT_DIR \
    --train_fps 16 \
    --samples_per_file 8 \
    --flush_frequency 8 \
    --video_length_tolerance_range 5 \
    --preprocess_task "ode_trajectory" 
