#!/bin/bash

GPU_NUM=1 # 2,4,8
MODEL_PATH="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
MODEL_TYPE="wan"
DATA_MERGE_PATH="data/crush-smol/merge.txt"
OUTPUT_DIR="data/crush-smol_processed_ti2v/"

torchrun --nproc_per_node=$GPU_NUM \
    fastvideo/pipelines/preprocess/v1_preprocess.py \
    --config examples/distill/Wan2.2-TI2V-5B-Diffusers/crush_smol/preprocess_wan_data_ti2v_5b.yaml \
    --model_path $MODEL_PATH \
    --preprocess.data_merge_path $DATA_MERGE_PATH \
    --preprocess.dataset_output_dir=$OUTPUT_DIR
