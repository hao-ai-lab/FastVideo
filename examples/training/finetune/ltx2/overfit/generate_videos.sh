#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/generate_videos.py"

# Configurable arguments
JSON_FILE="/home/d1su/codes/FastVideo-demo/data/Davids048/ltx2-data/merged_prompts_semantic_unique_16k/video_prompts.gpt-5-mini-2025-08-07.0-1000.jsonl"
START_IDX=${1:-0}
END_IDX=${2:-125}
OUTPUT_DIR="${SCRIPT_DIR}/data/generated_videos"
NUM_GPUS=1


# Logging
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/generate_videos_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "[INFO] JSON_FILE=${JSON_FILE}" | tee -a "${LOG_FILE}"
echo "[INFO] START_IDX=${START_IDX}" | tee -a "${LOG_FILE}"
echo "[INFO] END_IDX=${END_IDX}" | tee -a "${LOG_FILE}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "[INFO] NUM_GPUS=${NUM_GPUS}" | tee -a "${LOG_FILE}"
echo "[INFO] LOG_FILE=${LOG_FILE}" | tee -a "${LOG_FILE}"

python "${PY_SCRIPT}" \
  "${JSON_FILE}" \
  "${START_IDX}" \
  "${END_IDX}" \
  "${OUTPUT_DIR}" \
  --num-gpus "${NUM_GPUS}" \
  --skip-existing \
  2>&1 | tee -a "${LOG_FILE}"
