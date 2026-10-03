#!/bin/bash

GPU_NUM=1 # 2,4,8
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
MODEL_TYPE="wan"
DATA_MERGE_PATH="$(dirname "$0")/crush_smol_prompts.txt"
OUTPUT_DIR="data/crush-smol_processed_t2v_1_3b_ode_init/"

torchrun --nproc_per_node=$GPU_NUM \
    fastvideo/pipelines/preprocess/v1_preprocess.py \
    --config examples/training/consistency_finetune/causal_ode_init/preprocess_wan_data_crush_smol.yaml \
    --model_path $MODEL_PATH \
    --preprocess.data_merge_path $DATA_MERGE_PATH \
    --preprocess.dataset_output_dir=$OUTPUT_DIR
