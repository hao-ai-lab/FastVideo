#!/usr/bin/env bash
# Launch training from a YAML config.
#
# Usage:
#   bash examples/train/run.sh <config.yaml> [--dotted.key value ...]
#
# Examples:
#   bash examples/train/run.sh examples/train/finetune_wan2.1_t2v_1.3B_vsa_phase3.4_0.9sparsity.yaml
#   bash examples/train/run.sh examples/train/distill_wan2.1_t2v_1.3B_dmd2.yaml --dry-run
#   bash examples/train/run.sh examples/train/distill_wan2.1_t2v_1.3B_dmd2.yaml \
#       --training.distributed.num_gpus 4 \
#       --training.optimizer.learning_rate 1e-5
#   bash examples/train/run.sh examples/train/distill_wan2.1_t2v_1.3B_dmd2.yaml \
#       --training.checkpoint.resume_from_checkpoint outputs/my_run/checkpoint-1000
#
# Logs are written to logs/<config_name>_<timestamp>.log (and also printed to stdout).

set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml> [extra flags...]}"
shift

# ── GPU / node settings ──────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NUM_GPUS="${NUM_GPUS:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"
export TOKENIZERS_PARALLELISM=false
# ── W&B ──────────────────────────────────────────────────────────
export WANDB_API_KEY="${WANDB_API_KEY:-7ff8b6e8356924f7a6dd51a0342dd1a422ea9352}"
export WANDB_MODE="${WANDB_MODE:-online}"


# ── Log file ─────────────────────────────────────────────────────
CONFIG_NAME="$(basename "${CONFIG}" .yaml)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-examples/train}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${CONFIG_NAME}_${TIMESTAMP}.log"

set +u
source ~/conda/miniconda/bin/activate
conda activate mhuo-fv
set -u
export PYTHONPATH="/mnt/weka/home/hao.zhang/mhuo/FastVideo-refactor:${PYTHONPATH:-}"

echo "=== Train Training ==="
echo "Config:      ${CONFIG}"
echo "Num GPUs:    ${NUM_GPUS}"
echo "Num Nodes:   ${NNODES}"
echo "Node Rank:   ${NODE_RANK}"
echo "Master:      ${MASTER_ADDR}:${MASTER_PORT}"
echo "Extra args:  $*"
echo "Log file:    ${LOG_FILE}"
echo "=============================="

python -m torch.distributed.run \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --nproc_per_node "${NUM_GPUS}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    fastvideo/train/entrypoint/train.py \
    --config "${CONFIG}" \
    "$@" \
    2>&1 | tee "${LOG_FILE}"
