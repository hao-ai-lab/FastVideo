#!/bin/bash

GPU_NUM=2 # 2,4,8
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DATASET_PATH="data/crush-smol/"
OUTPUT_DIR="data/crush-smol_processed_t2v/"

torchrun --nproc_per_node=$GPU_NUM \
    --master_port=29513 \
    -m fastvideo.pipelines.preprocess.v1_preprocessing_new \
    --config examples/training/finetune/wan_t2v_1.3B/crush_smol/preprocess_wan_data_t2v_new.yaml \
    --model_path $MODEL_PATH \
    --preprocess.dataset_path $DATASET_PATH \
    --preprocess.dataset_output_dir $OUTPUT_DIR
