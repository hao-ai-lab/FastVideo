#!/bin/bash

GPU_NUM=4
MODEL_PATH="FastVideo/LTX2-Distilled-Diffusers"
DATASET_PATH="data/crush-smol"
OUTPUT_DIR="$DATASET_PATH"
WITH_AUDIO=true


torchrun  --nproc_per_node=$GPU_NUM \
    --master_port=29513 \
    -m fastvideo.pipelines.preprocess.v1_preprocessing_new \
    --config examples/training/finetune/ltx2/preprocess_ltx2_data_t2v_new.yaml \
    --model_path $MODEL_PATH \
    --preprocess.dataset_path $DATASET_PATH \
    --preprocess.dataset_output_dir $OUTPUT_DIR \
    --preprocess.with_audio $WITH_AUDIO
