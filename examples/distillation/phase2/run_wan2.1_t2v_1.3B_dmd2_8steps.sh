#!/bin/bash
set -euo pipefail
set -x

# NOTE:
# Phase 2 expects an explicit YAML path (we keep runnable YAML under outside/):
#   fastvideo/distillation/outside/fastvideo/configs/distillation/*.yaml

export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export TORCH_NCCL_ENABLE_MONITORING=${TORCH_NCCL_ENABLE_MONITORING:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export FASTVIDEO_ATTENTION_BACKEND=${FASTVIDEO_ATTENTION_BACKEND:-FLASH_ATTN}
export WANDB_MODE=${WANDB_MODE:-offline}
export MASTER_PORT=${MASTER_PORT:-29506}

CONFIG=${CONFIG:-"fastvideo/distillation/outside/fastvideo/configs/distillation/distill_wan2.1_t2v_1.3B_dmd2_8steps.yaml"}
if [[ ! -f "$CONFIG" ]]; then
  echo "Missing Phase 2 YAML config at: $CONFIG" >&2
  exit 1
fi
NUM_GPUS=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG', encoding='utf-8')); print(int(cfg.get('training', {}).get('num_gpus', 1) or 1))")

torchrun \
  --nnodes 1 \
  --master_port "$MASTER_PORT" \
  --nproc_per_node "$NUM_GPUS" \
  fastvideo/training/distillation.py \
  --config "$CONFIG"
