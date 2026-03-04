#!/bin/bash
set -e -x

# One-shot launch script for Phase 2 (YAML-only, standalone runtime) Wan DMD2
# few-step distillation.
#
# Uses the same defaults as Phase0/Phase1 temp.sh:
# - parquet dataset folder: data/Wan-Syn_77x448x832_600k
# - validation json: examples/training/finetune/Wan2.1-VSA/Wan-Syn-Data/validation_4.json
#
# Notes:
# - Phase 2 expects an explicit YAML path (YAML-only entrypoint).
#   We keep runnable YAML next to this script under:
#     examples/distillation/phase2/*.yaml
# - By default this runs W&B in offline mode (safer for overnight runs).
#   If you want online logging:
#     export WANDB_MODE=online
#     export WANDB_API_KEY=...

export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export TORCH_NCCL_ENABLE_MONITORING=${TORCH_NCCL_ENABLE_MONITORING:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export FASTVIDEO_ATTENTION_BACKEND=${FASTVIDEO_ATTENTION_BACKEND:-FLASH_ATTN}
export WANDB_BASE_URL=${WANDB_BASE_URL:-"https://api.wandb.ai"}
export WANDB_MODE=${WANDB_MODE:-offline}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-29507}

if [[ "$WANDB_MODE" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_MODE=online requires WANDB_API_KEY in env." >&2
  exit 1
fi

CONFIG=${CONFIG:-"examples/distillation/phase2/distill_wan2.1_t2v_1.3B_dmd2_8steps.yaml"}

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing Phase 2 YAML config at: $CONFIG" >&2
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
