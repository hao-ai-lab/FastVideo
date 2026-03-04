#!/usr/bin/env bash
# Launch distillation training from a v3 YAML config.
#
# Usage:
#   bash dev/refactor/run.sh <config.yaml> [extra torchrun/script flags]
#
# Examples:
#   # 8-GPU self-forcing run
#   bash dev/refactor/run.sh examples/distillation/refactor/self_forcing_wangame_causal_v3.yaml
#
#   # DFSFT with output dir override
#   bash dev/refactor/run.sh examples/distillation/refactor/dfsft_wangame_causal_v3.yaml \
#       --override-output-dir outputs/my_run
#
#   # Dry-run (parse config only, no training)
#   bash dev/refactor/run.sh examples/distillation/refactor/self_forcing_wangame_causal_v3.yaml --dry-run

set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml> [extra flags...]}"
shift

# ── GPU / node settings ──────────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NUM_GPUS="${NUM_GPUS:-1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "=== Distillation Training ==="
echo "Config:      ${CONFIG}"
echo "Num GPUs:    ${NUM_GPUS}"
echo "Num Nodes:   ${NNODES}"
echo "Node Rank:   ${NODE_RANK}"
echo "Master:      ${MASTER_ADDR}:${MASTER_PORT}"
echo "Extra args:  $*"
echo "=============================="

torchrun \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --nproc_per_node "${NUM_GPUS}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    fastvideo/training/distillation.py \
    --config "${CONFIG}" \
    "$@"
