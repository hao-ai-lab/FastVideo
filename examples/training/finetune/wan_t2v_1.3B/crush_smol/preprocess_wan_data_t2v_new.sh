#!/bin/bash

GPU_NUM=1 # 2,4,8
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
MODEL_TYPE="wan"
DATASET_PATH="data/crush-smol-test/"
OUTPUT_DIR="data/crush-smol_processed_t2v/"

torchrun --nproc_per_node=$GPU_NUM \
    fastvideo/pipelines/preprocess/v1_preprocessing_new.py \
    --model_path $MODEL_PATH \
    --mode preprocess \
    --workload_type t2v \
    --preprocess.dataset_path $DATASET_PATH \
    --preprocess.dataset_output_dir $OUTPUT_DIR \
    --preprocess.preprocess_video_batch_size 4 \
    --preprocess.dataloader_num_workers 0 \
    --preprocess.max_height 480 \
    --preprocess.max_width 832 \
    --preprocess.num_frames 77 \
    --preprocess.train_fps 16 \
    --preprocess.samples_per_file 8 \
    --preprocess.flush_frequency 8 \
    --preprocess.video_length_tolerance_range 5
