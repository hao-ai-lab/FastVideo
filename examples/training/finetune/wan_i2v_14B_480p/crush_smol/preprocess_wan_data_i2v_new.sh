#!/bin/bash

GPU_NUM=2 # 2,4,8
MODEL_PATH="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
DATASET_PATH="data/crush-smol/"
OUTPUT_DIR="data/crush-smol_processed_i2v/"

torchrun --nproc_per_node=$GPU_NUM \
    -m fastvideo.pipelines.preprocess.v1_preprocessing_new \
    --config examples/training/finetune/wan_i2v_14B_480p/crush_smol/preprocess_wan_data_i2v_new.yaml \
    --model_path $MODEL_PATH \
    --preprocess.dataset_path $DATASET_PATH \
    --preprocess.dataset_output_dir $OUTPUT_DIR
