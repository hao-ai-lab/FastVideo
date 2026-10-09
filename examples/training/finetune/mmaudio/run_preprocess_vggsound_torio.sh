#!/usr/bin/env bash

# Run the FastVideo MMAudio preprocessing pipeline with the optional torio
# reference media reader in an isolated environment.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  exec "${SCRIPT_DIR}/run_preprocess_vggsound.sh" "$@"
fi

TORIO_VENV_PATH="${TORIO_VENV_PATH:-${REPO_ROOT}/../envs/fastvideo-mmaudio-torio}"
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to the extracted VGGSound directory}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATASET_PATH}/mmaudio_features_torio}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"

if [[ ! -x "${TORIO_VENV_PATH}/bin/torchrun" ]]; then
  echo "FastVideo torio environment does not exist: ${TORIO_VENV_PATH}" >&2
  exit 1
fi

export PYTHONNOUSERSITE=1

VENV_PATH="${TORIO_VENV_PATH}" \
DATASET_PATH="${DATASET_PATH}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS}" \
VIDEO_LOADER_TYPE=torio \
  exec "${SCRIPT_DIR}/run_preprocess_vggsound.sh" "$@"
