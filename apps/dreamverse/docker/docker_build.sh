#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
IMAGE="${DREAMVERSE_IMAGE:-dreamverse:dev}"

build_args=()
[[ -n "${FLASH_ATTN_REF:-}"  ]] && build_args+=(--build-arg "FLASH_ATTN_REF=${FLASH_ATTN_REF}")
[[ -n "${FLASH_ATTN_REPO:-}" ]] && build_args+=(--build-arg "FLASH_ATTN_REPO=${FLASH_ATTN_REPO}")
[[ -n "${CUDA_TAG:-}"        ]] && build_args+=(--build-arg "CUDA_TAG=${CUDA_TAG}")
[[ -n "${BUILD_FASTVIDEO_KERNEL_FROM_SOURCE:-}" ]] && \
  build_args+=(--build-arg "BUILD_FASTVIDEO_KERNEL_FROM_SOURCE=${BUILD_FASTVIDEO_KERNEL_FROM_SOURCE}")

exec docker build \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t "${IMAGE}" \
  "${build_args[@]}" \
  "${REPO_ROOT}"
