#!/bin/bash
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --exclusive

# ===========================================================================
# Unified sbatch script for preparing raw + precomputed datasets.
#
# Usage:
#   sbatch --job-name=ltx2_raw_7000_8000 prepare_raw.sbatch 7000-8000
#
# Or submit a batch of ranges in one shot:
#   for r in 0-1000 1000-2000 2000-3000 3000-4000 4000-5000 5000-6000 \
#            6000-7000 7000-8000 8000-9000 9000-10000 10000-11000 \
#            11000-12000 12000-13000 13000-14000 14000-15000 15000-16000; do
#     sbatch --job-name="ltx2_raw_${r//-/_}" prepare_raw.sbatch "$r"
#   done
# ===========================================================================

set -euo pipefail

RANGE="${1:?Usage: sbatch --job-name=ltx2_raw_START_END prepare_raw.sbatch <START-END>}"

ROOT_DIR="/mnt/weka/home/hao.zhang/ds8-agent/codes/FastVideo-demo"
SCRIPT_DIR="${ROOT_DIR}/examples/training/finetune/ltx2/overfit_v2/launch_scripts"

mkdir -p "${SCRIPT_DIR}/slurm_logs"

JSONL="${ROOT_DIR}/data/Davids048/ltx2-data/merged_prompts_semantic_unique_64k/video_prompts.gpt-5-mini-2025-08-07.${RANGE}.jsonl"
CONFIG="${ROOT_DIR}/examples/training/finetune/ltx2/overfit_v2/configs/overfit_v2.yaml"
PREPARE_SCRIPT="${ROOT_DIR}/examples/training/finetune/ltx2/overfit_v2/prepare_raw_dataset.sh"
PRECOMPUTE_SCRIPT="${ROOT_DIR}/examples/training/finetune/ltx2/overfit_v2/precompute_dataset.sh"

if [[ ! -f "${JSONL}" ]]; then
  echo "ERROR: JSONL file not found: ${JSONL}" >&2
  exit 1
fi

DATA_ROOT_BASE="$(sed -n 's/^data_root:[[:space:]]*//p' "${CONFIG}" | head -n1 | sed -e 's/[[:space:]]*$//')"
if [[ "${DATA_ROOT_BASE}" != /* ]]; then
  DATA_ROOT_BASE="${ROOT_DIR}/${DATA_ROOT_BASE}"
fi
JSONL_STEM="$(basename "${JSONL}" .jsonl)"
OUTPUT_ROOT="${DATA_ROOT_BASE}/${JSONL_STEM}_output"

START_IDX=0
END_IDX=1000

echo "[INFO] RANGE=${RANGE}"
echo "[INFO] JSONL=${JSONL}"
echo "[INFO] OUTPUT_ROOT=${OUTPUT_ROOT}"

bash "${PREPARE_SCRIPT}" \
  --config "${CONFIG}" \
  --prompts-jsonl "${JSONL}" \
  --num-gpus 8 \
  --output-root "${OUTPUT_ROOT}" \
  --start-idx "${START_IDX}" \
  --end-idx "${END_IDX}"

bash "${PRECOMPUTE_SCRIPT}" \
  --config "${CONFIG}" \
  --data-root "${OUTPUT_ROOT}"
