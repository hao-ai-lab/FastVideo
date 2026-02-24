#!/bin/bash
set -euo pipefail
if [[ "${DEBUG:-0}" == "1" ]]; then
  set -x
fi

# One-shot launch script for Phase 3.2 (ODE/SDE sampler selection in `WanPipeline`).
#
# Compared to phase3_1, validation no longer needs `WanDMDPipeline`; the method
# supplies `sampler_kind + sampling_timesteps` via `ValidationRequest`, and
# `WanValidator` runs `WanPipeline` with the requested loop.

export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export TORCH_NCCL_ENABLE_MONITORING=${TORCH_NCCL_ENABLE_MONITORING:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export FASTVIDEO_ATTENTION_BACKEND=${FASTVIDEO_ATTENTION_BACKEND:-FLASH_ATTN}
export WANDB_BASE_URL=${WANDB_BASE_URL:-"https://api.wandb.ai"}
export WANDB_MODE=${WANDB_MODE:-offline}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-29512}

if [[ "$WANDB_MODE" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_MODE=online requires WANDB_API_KEY in env." >&2
  exit 1
fi

CONFIG=${CONFIG:-"fastvideo/distillation/outside/fastvideo/configs/distillation/distill_wan2.1_t2v_1.3B_dmd2_8steps_phase3.2.yaml"}

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing distillation YAML config at: $CONFIG" >&2
  exit 1
fi

NUM_GPUS=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG', encoding='utf-8')); print(int(cfg.get('training', {}).get('num_gpus', 1) or 1))")

torchrun \
  --nnodes 1 \
  --nproc_per_node "$NUM_GPUS" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  fastvideo/training/distillation.py \
  --config "$CONFIG"

