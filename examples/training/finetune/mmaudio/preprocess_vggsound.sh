#!/usr/bin/env bash

set -euo pipefail

GPU_NUM="${GPU_NUM:-4}"
MODEL_PATH="${MODEL_PATH:-converted_weights/mmaudio/preprocess_44k}"
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to the extracted VGGSound directory}"
SPLIT="${SPLIT:-train}"
CAPTION_PATH="${CAPTION_PATH:-${DATASET_PATH}/sets/filtered_caption/vgg-${SPLIT}-filtered-caption.tsv}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATASET_PATH}/mmaudio_features/${SPLIT}}"
MASTER_PORT="${MASTER_PORT:-29513}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
SAMPLES_PER_FILE="${SAMPLES_PER_FILE:-256}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

if [[ ! -f "${CAPTION_PATH}" ]]; then
  echo "Caption manifest does not exist: ${CAPTION_PATH}" >&2
  exit 1
fi

torchrun --standalone \
  --nproc_per_node="${GPU_NUM}" \
  --master_port="${MASTER_PORT}" \
  -m fastvideo.pipelines.preprocess.v1_preprocessing_new \
  --model-path "${MODEL_PATH}" \
  --mode preprocess \
  --workload-type v2a \
  --preprocess.dataset-type vggsound \
  --preprocess.dataset-path "${DATASET_PATH}" \
  --preprocess.dataset-metadata-path "${CAPTION_PATH}" \
  --preprocess.dataset-split "${SPLIT}" \
  --preprocess.dataset-output-dir "${OUTPUT_DIR}" \
  --preprocess.preprocess-video-batch-size "${BATCH_SIZE}" \
  --preprocess.dataloader-num-workers "${DATALOADER_NUM_WORKERS}" \
  --preprocess.samples-per-file "${SAMPLES_PER_FILE}"
