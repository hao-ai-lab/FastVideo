#!/usr/bin/env bash
# Capture one FastH3 Preview generation with CUDA and FastVideo NVTX ranges.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${WORKTREE_ROOT}/.." && pwd)"
PYTHON_BIN="${WORKSPACE_ROOT}/.venv-fv/bin/python"
NSYS_BIN=/usr/local/cuda/bin/nsys
PROFILE_SCRIPT="${SCRIPT_DIR}/basic_fasth3_simplified_profile.py"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export FASTVIDEO_FA4=1
export FASTVIDEO_INFERENCE_TORCH_COMPILE=0
export FASTVIDEO_MINIMAX_H3_FUSIONS=all
export FASTVIDEO_NVTX_PROFILE=1
export FASTVIDEO_VSA_SM100A=0
export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKTREE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PROMPT="integrated_multimodal_description: A red fox runs through fresh snow at dawn. overall_soundscape: Fast pawsteps in snow, winter wind, and distant birds."

# Options: dense-datafree, vsa-datafree, vsa-synthetic-step1300, vsa-synthetic-step1900
VARIANT="vsa-datafree"
NUM_FRAMES=345
WARMUP_RUNS=3

PROFILE_ID="${VARIANT}_4gpu_sp4_${NUM_FRAMES}frames"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RESULT_DIR="${WORKTREE_ROOT}/runs/fasth3_lora_preview_profile/${PROFILE_ID}/${RUN_ID}"
MEDIA_DIR="${RESULT_DIR}/media"
RUN_LOG="${RESULT_DIR}/run.log"
mkdir -p "${MEDIA_DIR}"

"${NSYS_BIN}" profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --output="${RESULT_DIR}/${PROFILE_ID}" \
  "${PYTHON_BIN}" "${PROFILE_SCRIPT}" \
  "${VARIANT}" \
  --num-frames "${NUM_FRAMES}" \
  --warmup-runs "${WARMUP_RUNS}" \
  --prompt "${PROMPT}" \
  --output "${MEDIA_DIR}" \
  2>&1 | tee "${RUN_LOG}"

printf 'RESULT_DIR=%s\n' "${RESULT_DIR}"
printf 'NSYS_REPORT=%s\n' "${RESULT_DIR}/${PROFILE_ID}.nsys-rep"
printf 'MEDIA_DIR=%s\n' "${MEDIA_DIR}"
