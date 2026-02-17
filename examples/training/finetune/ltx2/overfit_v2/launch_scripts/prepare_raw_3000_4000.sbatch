#!/bin/bash
#SBATCH --job-name=ltx2_raw_3000_4000
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --exclusive

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"

mkdir -p "${SCRIPT_DIR}/slurm_logs"

JSONL="${ROOT_DIR}/data/Davids048/ltx2-data/merged_prompts_semantic_unique_64k/video_prompts.gpt-5-mini-2025-08-07.3000-4000.jsonl"
CONFIG="${ROOT_DIR}/examples/training/finetune/ltx2/overfit_v2/configs/overfit_v2.yaml"
PREPARE_SCRIPT="${ROOT_DIR}/examples/training/finetune/ltx2/overfit_v2/prepare_raw_dataset.sh"
PRECOMPUTE_SCRIPT="${ROOT_DIR}/examples/training/finetune/ltx2/overfit_v2/precompute_dataset.sh"

DATA_ROOT_BASE="$(sed -n 's/^data_root:[[:space:]]*//p' "${CONFIG}" | head -n1 | sed -e 's/[[:space:]]*$//')"
if [[ "${DATA_ROOT_BASE}" != /* ]]; then
  DATA_ROOT_BASE="${ROOT_DIR}/${DATA_ROOT_BASE}"
fi
JSONL_STEM="$(basename "${JSONL}" .jsonl)"
OUTPUT_ROOT="${DATA_ROOT_BASE}/${JSONL_STEM}_output"

bash "${PREPARE_SCRIPT}" \
  --config "${CONFIG}" \
  --prompts-jsonl "${JSONL}" \
  --num-gpus 8 \
  --output-root "${OUTPUT_ROOT}"

bash "${PRECOMPUTE_SCRIPT}" \
  --config "${CONFIG}" \
  --data-root "${OUTPUT_ROOT}"
