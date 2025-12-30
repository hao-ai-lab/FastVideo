#!/bin/bash

GPU_NUM=1 # 2,4,8
MODEL_PATH="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
DATASET_PATH="data/crush-smol"
OUTPUT_DIR="data/crush-smol_processed_t2v_hunyuan15/"

torchrun --nproc_per_node=$GPU_NUM --master_port=29513 \
    -m fastvideo.pipelines.preprocess.v1_preprocessing_new \
    --model_path $MODEL_PATH \
    --mode preprocess \
    --workload_type t2v \
    --preprocess.video_loader_type torchvision \
    --preprocess.dataset_type merged \
    --preprocess.dataset_path $DATASET_PATH \
    --preprocess.dataset_output_dir $OUTPUT_DIR \
    --preprocess.preprocess_video_batch_size 2 \
    --preprocess.dataloader_num_workers 0 \
    --preprocess.max_height 480 \
    --preprocess.max_width 832 \
    --preprocess.num_frames 61 \
    --preprocess.train_fps 15 \
    --preprocess.samples_per_file 8 \
    --preprocess.flush_frequency 8 \
    --preprocess.video_length_tolerance_range 5
