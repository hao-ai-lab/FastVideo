#!/usr/bin/env bash
set -euo pipefail

: "${CEREBRAS_API_KEY:?CEREBRAS_API_KEY not set on host}"
: "${GROQ_API_KEY:?GROQ_API_KEY not set on host}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DREAMVERSE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${DREAMVERSE_IMAGE:-dreamverse:dev}"
PORT="${BACKEND_PORT:-8009}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
OUTPUTS_DIR="${DREAMVERSE_OUTPUTS_DIR:-${DREAMVERSE_ROOT}/outputs}"
GPU_REQUEST="${DREAMVERSE_DOCKER_GPUS:-all}"

mkdir -p "${HF_CACHE}" "${OUTPUTS_DIR}"

env_args=(
  -e "CEREBRAS_API_KEY=${CEREBRAS_API_KEY}"
  -e "GROQ_API_KEY=${GROQ_API_KEY}"
)
[[ -n "${ENABLE_TORCH_COMPILE:-}" ]] && env_args+=(-e "ENABLE_TORCH_COMPILE=${ENABLE_TORCH_COMPILE}")
[[ -n "${FASTVIDEO_GPU_COUNT:-}" ]] && env_args+=(-e "FASTVIDEO_GPU_COUNT=${FASTVIDEO_GPU_COUNT}")

exec docker run --rm --gpus "${GPU_REQUEST}" --init \
  -p "${PORT}:8009" \
  "${env_args[@]}" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "${OUTPUTS_DIR}:/var/lib/dreamverse/outputs" \
  "${IMAGE}"
