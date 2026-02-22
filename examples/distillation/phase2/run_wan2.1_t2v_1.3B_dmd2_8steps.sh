#!/bin/bash
set -euo pipefail
set -x

export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export TORCH_NCCL_ENABLE_MONITORING=${TORCH_NCCL_ENABLE_MONITORING:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export FASTVIDEO_ATTENTION_BACKEND=${FASTVIDEO_ATTENTION_BACKEND:-FLASH_ATTN}
export WANDB_MODE=${WANDB_MODE:-offline}
export MASTER_PORT=${MASTER_PORT:-29506}

NUM_GPUS=${NUM_GPUS:-1}
CONFIG=${CONFIG:-"examples/distillation/phase2/distill_wan2.1_t2v_1.3B_dmd2_8steps.yaml"}

torchrun \
  --nnodes 1 \
  --master_port "$MASTER_PORT" \
  --nproc_per_node "$NUM_GPUS" \
  fastvideo/training/distillation.py \
  --config "$CONFIG"

