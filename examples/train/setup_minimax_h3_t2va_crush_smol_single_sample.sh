#!/usr/bin/env bash
# Download the pinned Crush-Smol dataset and encode one H3 T2VA training record.

set -euo pipefail
set +x

WORKTREE_ROOT=/mnt/weka/shrd/wm/junda/fv-hub/fastvideo-add-h3-sft
ENV_FILE=/mnt/weka/shrd/wm/junda/fv-hub/.env
VENV_BIN=/mnt/weka/shrd/wm/junda/fv-hub/.venv/bin
DATASET_REVISION=1a850a74e92d5ac3daa273ea658ec60e92fbaf4e
DATASET_DIR="${WORKTREE_ROOT}/data/crush-smol"
MODEL_INDEX="${WORKTREE_ROOT}/data/models/MiniMax-H3/model_index.json"
OUTPUT_PARQUET="${WORKTREE_ROOT}/data/crush-smol_h3_t2va_single_sample_preprocessed/data_00000.parquet"

[[ -f "${ENV_FILE}" ]] || { echo "Missing environment file: ${ENV_FILE}" >&2; exit 1; }
[[ -f "${MODEL_INDEX}" ]] || { echo "Missing MiniMax H3 checkpoint: ${MODEL_INDEX}" >&2; exit 1; }
[[ -x "${VENV_BIN}/hf" ]] || { echo "Missing Hugging Face CLI: ${VENV_BIN}/hf" >&2; exit 1; }
[[ -x "${VENV_BIN}/torchrun" ]] || { echo "Missing preprocessing environment: ${VENV_BIN}" >&2; exit 1; }

set -a
source "${ENV_FILE}"
set +a

cd "${WORKTREE_ROOT}"
"${VENV_BIN}/hf" download wlsaidhi/crush-smol-merged \
    --repo-type dataset \
    --revision "${DATASET_REVISION}" \
    --local-dir "${DATASET_DIR}"

CUDA_VISIBLE_DEVICES=0 "${VENV_BIN}/torchrun" \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=1 \
    -m fastvideo.pipelines.preprocess.preprocess_minimax_h3_overfit

[[ -f "${OUTPUT_PARQUET}" ]] || { echo "Preprocessing did not create ${OUTPUT_PARQUET}" >&2; exit 1; }
echo "Prepared one Crush-Smol H3 training record at ${OUTPUT_PARQUET}"
