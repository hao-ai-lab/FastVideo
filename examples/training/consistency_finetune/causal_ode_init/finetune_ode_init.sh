#!/bin/bash

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers"
DATA_DIR="data/crush-smol_processed_t2v_1_3b_ode_init/"
VALIDATION_DATASET_FILE="$(dirname "$0")/validation.json"
NUM_GPUS=1
# IP=[MASTER NODE IP]

# If you do not have 32 GPUs and to fit in memory, you can: 1. increase sp_size. 2. reduce num_latent_t
torchrun \
  --nnodes 1 \
  --nproc_per_node $NUM_GPUS \
    fastvideo/training/ode_causal_pipeline.py \
    --config examples/training/consistency_finetune/causal_ode_init/finetune_ode_init.yaml \
    --model_path "$MODEL_PATH" \
    --engine.num_gpus "$NUM_GPUS" \
    --engine.parallelism.hsdp_shard_dim "$NUM_GPUS" \
    --training.data.data_path "$DATA_DIR"
