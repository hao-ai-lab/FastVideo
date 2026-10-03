#!/bin/bash

GPU_NUM=1 # 2,4,8
MODEL_PATH="weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers"
MODEL_TYPE="wan"
DATA_MERGE_PATH="data/crush-smol/merge.txt"
OUTPUT_DIR="data/crush-smol_processed_i2v_1_3b_inp/"

torchrun --nproc_per_node=$GPU_NUM \
    fastvideo/pipelines/preprocess/v1_preprocess.py \
    --config examples/training/finetune/Wan2.1-Fun-1.3B-InP/crush_smol/preprocess_wan_data_i2v.yaml \
    --model_path $MODEL_PATH \
    --preprocess.data_merge_path $DATA_MERGE_PATH \
    --preprocess.dataset_output_dir=$OUTPUT_DIR