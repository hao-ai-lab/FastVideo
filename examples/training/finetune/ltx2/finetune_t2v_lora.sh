#!/bin/bash

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="FastVideo/LTX2-Distilled-Diffusers"
DATA_DIR="data/crush-smol"
VALIDATION_DATASET_FILE="$(dirname "$0")/validation.json"
NUM_GPUS=1

torchrun \
  --nnodes 1 \
  --nproc_per_node $NUM_GPUS \
    fastvideo/training/ltx2_training_pipeline.py \
    --config examples/training/finetune/ltx2/finetune_t2v_lora.yaml \
    --model_path "$MODEL_PATH" \
    --engine.num_gpus "$NUM_GPUS" \
    --engine.parallelism.sp_size "$NUM_GPUS" \
    --engine.parallelism.hsdp_shard_dim "$NUM_GPUS" \
    --training.data.data_path "$DATA_DIR" \
    --training.validation.dataset_file "$VALIDATION_DATASET_FILE"
