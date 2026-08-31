#!/bin/bash
# Launch an Nsight Systems profile for the compiled MiniMax H3 pipeline.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${WORKTREE_ROOT}/.." && pwd)"
PYTHON_BIN="${WORKSPACE_ROOT}/.venv-fv/bin/python"
NSYS_BIN=/usr/local/cuda/bin/nsys
PROFILE_SCRIPT="${SCRIPT_DIR}/basic_minimax_h3_t2v_profile.py"
MODEL_PATH=MiniMaxAI/MiniMax-H3

export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
export FASTVIDEO_FA4=1
export FASTVIDEO_NVTX_PROFILE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORKTREE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

WARMUP_RUNS="${WARMUP_RUNS:-3}"
NUM_GPUS="${NUM_GPUS:-1}"
NUM_FRAMES=243
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

PROFILE_ID="fa4_${NUM_GPUS}gpu_sp${NUM_GPUS}_${NUM_FRAMES}frames"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RESULT_DIR="${WORKTREE_ROOT}/runs/minimax_h3_t2v_profile/${PROFILE_ID}/${RUN_ID}"
MEDIA_DIR="${RESULT_DIR}/media"
RUN_LOG="${RESULT_DIR}/run.log"
PROMPT="A small red panda walks through a misty bamboo forest at sunrise while soft wind moves the leaves."
mkdir -p "${MEDIA_DIR}"

"${NSYS_BIN}" profile \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --output="${RESULT_DIR}/${PROFILE_ID}" \
    "${PYTHON_BIN}" "${PROFILE_SCRIPT}" \
    --model-path "${MODEL_PATH}" \
    --num-gpus "${NUM_GPUS}" \
    --num-frames "${NUM_FRAMES}" \
    --steps 30 \
    --warmup-runs "${WARMUP_RUNS}" \
    --prompt "${PROMPT}" \
    --output "${MEDIA_DIR}" \
    2>&1 | tee "${RUN_LOG}"

printf 'RESULT_DIR=%s\n' "${RESULT_DIR}"
printf 'NSYS_REPORT=%s\n' "${RESULT_DIR}/${PROFILE_ID}.nsys-rep"
printf 'MEDIA_DIR=%s\n' "${MEDIA_DIR}"
