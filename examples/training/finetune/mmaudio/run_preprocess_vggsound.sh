#!/usr/bin/env bash

# One-command VGGSound feature extraction for native FastVideo MMAudio training.
# Every setting can be overridden with an environment variable; the defaults
# match the four-GPU GB200 machine used for the validated smoke tests.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  ./examples/training/finetune/mmaudio/run_preprocess_vggsound.sh

Optional environment variables:
  SPLITS="train val test"  Dataset splits to process sequentially
  GPU_NUM=4                Number of GPUs/ranks
  CUDA_VISIBLE_DEVICES=0,1,2,3
  BATCH_SIZE=16            Per-GPU batch size
  DATALOADER_NUM_WORKERS=16 Per-GPU DataLoader workers
  DATASET_PATH=...         Extracted VGGSound root
  MODEL_PATH=...           Converted MMAudio preprocessing model
  OUTPUT_ROOT=...          Root for split feature caches
  SAMPLES_PER_FILE=256     Samples per TensorDict shard
  MASTER_PORT=29513        First torchrun rendezvous port

Examples:
  SPLITS=train ./examples/training/finetune/mmaudio/run_preprocess_vggsound.sh
  BATCH_SIZE=8 GPU_NUM=2 CUDA_VISIBLE_DEVICES=0,1 \
    ./examples/training/finetune/mmaudio/run_preprocess_vggsound.sh
EOF
  exit 0
fi

if [[ $# -ne 0 ]]; then
  echo "Unknown argument: $1 (use --help)" >&2
  exit 2
fi

VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "FastVideo virtual environment does not exist: ${VENV_PATH}" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

GPU_NUM="${GPU_NUM:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-16}"
SAMPLES_PER_FILE="${SAMPLES_PER_FILE:-256}"
MASTER_PORT="${MASTER_PORT:-29513}"
SPLITS="${SPLITS:-train val test}"
DATASET_PATH="${DATASET_PATH:-/mnt/lustre/vlm-kai/datasets/VGGSound}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/converted_weights/mmaudio/preprocess_44k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATASET_PATH}/mmaudio_features}"

if [[ ! -d "${DATASET_PATH}/videos" ]]; then
  echo "VGGSound video directory does not exist: ${DATASET_PATH}/videos" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/model_index.json" ]]; then
  echo "Converted preprocessing model does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

echo "FastVideo MMAudio VGGSound preprocessing"
echo "  repository: ${REPO_ROOT}"
echo "  dataset:    ${DATASET_PATH}"
echo "  model:      ${MODEL_PATH}"
echo "  output:     ${OUTPUT_ROOT}"
echo "  splits:     ${SPLITS}"
echo "  GPUs:       ${CUDA_VISIBLE_DEVICES} (${GPU_NUM} ranks)"
echo "  batch:      ${BATCH_SIZE} per GPU"
echo "  workers:    ${DATALOADER_NUM_WORKERS} per GPU"

cd "${REPO_ROOT}"

port="${MASTER_PORT}"
for split in ${SPLITS}; do
  caption_path="${DATASET_PATH}/sets/filtered_caption/vgg-${split}-filtered-caption.tsv"
  output_dir="${OUTPUT_ROOT}/${split}"
  if [[ ! -f "${caption_path}" ]]; then
    echo "Caption manifest does not exist: ${caption_path}" >&2
    exit 1
  fi

  echo
  echo "===== Starting ${split} split ====="
  echo "Progress is reported below in batches; existing cached IDs are skipped."

  GPU_NUM="${GPU_NUM}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS}" \
  SAMPLES_PER_FILE="${SAMPLES_PER_FILE}" \
  MASTER_PORT="${port}" \
  MODEL_PATH="${MODEL_PATH}" \
  DATASET_PATH="${DATASET_PATH}" \
  CAPTION_PATH="${caption_path}" \
  OUTPUT_DIR="${output_dir}" \
  SPLIT="${split}" \
    bash "${SCRIPT_DIR}/preprocess_vggsound.sh"

  echo "===== Finished ${split}: ${output_dir} ====="
  port="$((port + 1))"
done

echo
echo "All requested VGGSound splits finished successfully."
