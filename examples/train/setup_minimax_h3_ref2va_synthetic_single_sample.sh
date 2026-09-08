#!/usr/bin/env bash
# Encode a caller-supplied manifest (or the bundled synthetic fixture) into
# MiniMax H3 Ref2VA training data.

set -euo pipefail
set +x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ENV_FILE="${ENV_FILE:-}"
FIXTURE_DIR="${REPO_ROOT}/examples/training/finetune/minimax-h3/synthetic"
MANIFEST="${MANIFEST:-${FIXTURE_DIR}/train.jsonl}"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/data/models/MiniMax-H3}"
MODEL_INDEX="${MODEL_DIR}/model_index.json"
TRANSFORMER_REF_DIR="${MODEL_DIR}/transformer_ref"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/data/synthetic_h3_ref2va_single_sample_preprocessed}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
REPLACE_EXISTING="${REPLACE_EXISTING:-0}"

command -v "${PYTHON_BIN}" >/dev/null || { echo "Python executable not found: ${PYTHON_BIN}" >&2; exit 1; }
"${PYTHON_BIN}" "${FIXTURE_DIR}/generate_fixture.py" --verify
[[ -f "${MANIFEST}" ]] || { echo "Missing MiniMax H3 Ref2VA manifest: ${MANIFEST}" >&2; exit 1; }
[[ -f "${MODEL_INDEX}" ]] || { echo "Missing MiniMax H3 checkpoint: ${MODEL_INDEX}" >&2; exit 1; }
[[ -d "${TRANSFORMER_REF_DIR}" ]] || { echo "Missing MiniMax H3 transformer_ref: ${TRANSFORMER_REF_DIR}" >&2; exit 1; }

if [[ -n "${ENV_FILE}" ]]; then
    [[ -f "${ENV_FILE}" ]] || { echo "Optional environment file not found: ${ENV_FILE}" >&2; exit 1; }
    set -a
    source "${ENV_FILE}"
    set +a
fi

replace_args=()
if [[ "${REPLACE_EXISTING}" == "1" ]]; then
    replace_args+=(--replace-existing)
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" \
    -m fastvideo.pipelines.preprocess.preprocess_minimax_h3_ref2va \
    --manifest "${MANIFEST}" \
    --validate-manifest-only

export CUDA_VISIBLE_DEVICES
"${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=1 \
    -m fastvideo.pipelines.preprocess.preprocess_minimax_h3_ref2va \
    --manifest "${MANIFEST}" \
    --model-path "${MODEL_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    "${replace_args[@]}"

"${PYTHON_BIN}" \
    -m fastvideo.pipelines.preprocess.preprocess_minimax_h3_ref2va \
    --manifest "${MANIFEST}" \
    --output-dir "${OUTPUT_DIR}" \
    --validate-only

echo "Prepared MiniMax H3 Ref2VA training data in ${OUTPUT_DIR}"
