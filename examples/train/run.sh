#!/usr/bin/env bash
# Launch training from a YAML config.
#
# Usage:
#   bash examples/train/run.sh <config.yaml> [--dotted.key value ...]
#
# Examples:
#   bash examples/train/run.sh examples/train/configs/fine_tuning/wan/t2v.yaml
#   bash examples/train/run.sh examples/train/configs/distribution_matching/wan/dmd2_t2v.yaml --dry-run
#   bash examples/train/run.sh examples/train/configs/distribution_matching/wan/dmd2_t2v.yaml \
#       --training.distributed.num_gpus 4 \
#       --training.optimizer.learning_rate 1e-5
#   bash examples/train/run.sh examples/train/configs/distribution_matching/wan/dmd2_t2v.yaml \
#       --training.checkpoint.resume_from_checkpoint outputs/my_run/checkpoint-1000
#
# Logs are written to logs/<config_name>_<timestamp>.log (and also printed to stdout).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG="${1:?Usage: $0 <config.yaml> [extra flags...]}"
shift

if [[ "${CONFIG}" != /* ]]; then
    CONFIG="$(pwd)/${CONFIG}"
fi

# ── GPU / node settings ──────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NUM_GPUS="${NUM_GPUS:-1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"
# ── W&B ──────────────────────────────────────────────────────────
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE="${WANDB_MODE:-online}"

# ── Log file ─────────────────────────────────────────────────────
CONFIG_NAME="$(basename "${CONFIG}" .yaml)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-examples/train}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${CONFIG_NAME}_${TIMESTAMP}.log"

echo "=== Train Training ==="
echo "Config:      ${CONFIG}"
echo "Num GPUs:    ${NUM_GPUS}"
echo "Num Nodes:   ${NNODES}"
echo "Node Rank:   ${NODE_RANK}"
echo "Master:      ${MASTER_ADDR}:${MASTER_PORT}"
echo "Extra args:  $*"
echo "Log file:    ${LOG_FILE}"
echo "Run log xlsx:${TRAINING_RUN_LOG_XLSX:-<disabled>}"
echo "=============================="

START_TIME="$(date +%s)"

set +e
python -m torch.distributed.run \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --nproc_per_node "${NUM_GPUS}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    -m fastvideo.train.entrypoint.train \
    --config "${CONFIG}" \
    "$@" \
    2>&1 | tee "${LOG_FILE}"
TRAIN_EXIT="${PIPESTATUS[0]}"
set -e

END_TIME="$(date +%s)"
WALL_TIME_SEC="$((END_TIME - START_TIME))"

RECORD_JSON="/tmp/fastvideo_training_run_record_${TIMESTAMP}.json"
python scripts/training/build_run_record.py \
    --config "${CONFIG}" \
    --log-file "${LOG_FILE}" \
    --exit-code "${TRAIN_EXIT}" \
    --wall-time-sec "${WALL_TIME_SEC}" \
    --output-json "${RECORD_JSON}" \
    --owner "${TRAINING_RUN_LOG_OWNER:-${USER:-}}"

python -c '
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    record = json.load(f)
parts = [
    "status={}".format(record.get("status", "")),
    "final_train_loss={}".format(record.get("final_train_loss", "")),
    "avg_step_time_sec={}".format(record.get("avg_step_time_sec", "")),
    "peak_vram_gb={}".format(record.get("peak_vram_gb", "")),
    "wall_time_hours={}".format(record.get("wall_time_hours", "")),
]
print("TRAINING_RUN_RECORD " + ", ".join(parts))
' "${RECORD_JSON}" | tee -a "${LOG_FILE}"

if [[ -n "${TRAINING_RUN_LOG_XLSX:-}" ]]; then
    python scripts/training/append_run_log.py \
        --workbook "${TRAINING_RUN_LOG_XLSX}" \
        --record "${RECORD_JSON}" \
        --sheet "${TRAINING_RUN_LOG_SHEET:-train_log}" | tee -a "${LOG_FILE}"
fi

exit "${TRAIN_EXIT}"
