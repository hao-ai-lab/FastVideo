#!/usr/bin/env bash
# Preprocess images for SD3.5 fine-tuning.
# Runs VAE encoding + CLIP-L/G + T5-XXL text encoding and writes parquet files.
#
# Usage:
#   DATA_MERGE_PATH=data/my_dataset/merge.txt \
#   OUTPUT_DIR=data/my_dataset_preprocessed \
#   bash scripts/preprocess/v1_preprocess_sd35_data.sh

GPU_NUM=1
MODEL_PATH="stabilityai/stable-diffusion-3.5-medium"
DATA_MERGE_PATH=${DATA_MERGE_PATH:-"data/sd35_dataset/merge.txt"}
OUTPUT_DIR=${OUTPUT_DIR:-"data/sd35_dataset_preprocessed"}

torchrun --nproc_per_node=$GPU_NUM \
    fastvideo/pipelines/preprocess/v1_preprocess.py \
    --model_path "$MODEL_PATH" \
    --data_merge_path "$DATA_MERGE_PATH" \
    --preprocess_video_batch_size 1 \
    --max_height 1024 \
    --max_width 1024 \
    --num_frames 1 \
    --dataloader_num_workers 0 \
    --output_dir="$OUTPUT_DIR" \
    --train_fps 1 \
    --samples_per_file 512 \
    --flush_frequency 512 \
    --preprocess_task "sd35"
