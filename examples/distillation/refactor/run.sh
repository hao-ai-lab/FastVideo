#!/usr/bin/env bash
# Launch distillation training from a v3 YAML config.
#
# Usage:
#   bash examples/distillation/refactor/run.sh <config.yaml> [extra flags]
#
# Examples:
#   bash examples/distillation/refactor/run.sh examples/distillation/refactor/self_forcing_wangame_causal_v3.yaml
#   bash examples/distillation/refactor/run.sh examples/distillation/refactor/dfsft_wangame_causal_v3.yaml --dry-run
#   bash examples/distillation/refactor/run.sh examples/distillation/refactor/dfsft_wangame_causal_v3.yaml \
#       --override-output-dir outputs/my_run
#
# Logs are written to logs/<config_name>_<timestamp>.log (and also printed to stdout).

set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml> [extra flags...]}"
shift

# ── GPU / node settings ──────────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NUM_GPUS="${NUM_GPUS:-1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"

# ── W&B ──────────────────────────────────────────────────────────────
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE="${WANDB_MODE:-online}"

# ── Log file ─────────────────────────────────────────────────────────
CONFIG_NAME="$(basename "${CONFIG}" .yaml)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-examples/distillation/refactor}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${CONFIG_NAME}_${TIMESTAMP}.log"

source ~/conda/miniconda/bin/activate
conda activate alexfv

echo "=== Distillation Training ==="
echo "Config:      ${CONFIG}"
echo "Num GPUs:    ${NUM_GPUS}"
echo "Num Nodes:   ${NNODES}"
echo "Node Rank:   ${NODE_RANK}"
echo "Master:      ${MASTER_ADDR}:${MASTER_PORT}"
echo "Extra args:  $*"
echo "Log file:    ${LOG_FILE}"
echo "=============================="

torchrun \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --nproc_per_node "${NUM_GPUS}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    fastvideo/training/distillation.py \
    --config "${CONFIG}" \
    "$@" \
    2>&1 | tee "${LOG_FILE}"
