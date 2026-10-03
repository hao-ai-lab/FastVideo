#!/bin/bash

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="FastVideo/LTX2-Distilled-Diffusers"
DATA_DIR="data/crush-smol"
VALIDATION_DATASET_FILE="$(dirname "$0")/validation.json"
echo  VALIDATION_DATASET_FILE: $VALIDATION_DATASET_FILE
NUM_GPUS=4
HEIGHT=1088
WIDTH=1920
FRAMES=121

# NOTE: Setting this environment variable to TORCH_SDPA to avoid the issue of stacking that failed in flash attn. 


torchrun \
  --nnodes 1 \
  --master_port 29501 \
  --nproc_per_node $NUM_GPUS \
    fastvideo/training/ltx2_training_pipeline.py \
    --config examples/training/finetune/ltx2/finetune_t2v.yaml \
    --model_path "$MODEL_PATH" \
    --engine.num_gpus "$NUM_GPUS" \
    --engine.parallelism.hsdp_shard_dim "$NUM_GPUS" \
    --training.data.data_path "$DATA_DIR" \
    --training.data.num_height "$HEIGHT" \
    --training.data.num_width "$WIDTH" \
    --training.data.num_frames "$FRAMES" \
    --training.validation.dataset_file "$VALIDATION_DATASET_FILE"
