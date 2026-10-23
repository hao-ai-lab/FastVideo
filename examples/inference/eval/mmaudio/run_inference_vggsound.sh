#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"
source .venv/bin/activate

MODEL_PATH="${MODEL_PATH:-converted_weights/mmaudio/small_44k_ema_300000}"
FEATURE_ROOT="${FEATURE_ROOT:-/mnt/lustre/vlm-kai/datasets/VGGSound/mmaudio_features_torio/test}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mmaudio_small_44k_ema_300000_vggsound_test}"
NUM_GPUS="${NUM_GPUS:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
COMPILE="${COMPILE:-1}"
MASTER_PORT="${MASTER_PORT:-29513}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

EXTRA_ARGS=()
if [[ "${MAX_SAMPLES}" -gt 0 ]]; then
  EXTRA_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ "${COMPILE}" == "1" ]]; then
  EXTRA_ARGS+=(--compile)
fi

torchrun \
  --standalone \
  --nproc_per_node="${NUM_GPUS}" \
  --master_port="${MASTER_PORT}" \
  examples/inference/eval/mmaudio/eval_mmaudio_dataset.py \
  --model-path "${MODEL_PATH}" \
  --feature-root "${FEATURE_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --duration-seconds 8 \
  --num-inference-steps 25 \
  --guidance-scale 4.5 \
  --seed 14159265 \
  --num-workers "${NUM_WORKERS}" \
  "${EXTRA_ARGS[@]}"

echo "Generated audio: ${OUTPUT_DIR}/audio"
echo "Evaluation manifest: ${OUTPUT_DIR}/eval_manifest.jsonl"
