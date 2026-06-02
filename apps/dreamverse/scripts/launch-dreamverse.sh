#!/usr/bin/env bash
# launch-dreamverse.sh — launch dreamverse-server on a compute node.
#
# Usage (from repo root):
#   CUDA_VISIBLE_DEVICES=0 bash apps/dreamverse/scripts/launch-dreamverse.sh

set -euo pipefail

CONDA_PREFIX="$HOME/miniconda3/envs/dreamverse"
CUDA_RT_DIR="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib"
GXX="$CONDA_PREFIX/bin/aarch64-conda-linux-gnu-g++"


export CUDA_HOME="$CONDA_PREFIX"
export FASTVIDEO_ENABLE_STARTUP_WARMUP=true
export FASTVIDEO_ENABLE_PROMPT_SAFETY=true
export DREAMVERSE_MAX_AUTOTUNE=true
export LTX2_USE_DISTILLED_SIGMAS=0
export DREAMVERSE_SESSION_TIMEOUT_SECONDS=1800
export CEREBRAS_API_KEY="${CEREBRAS_API_KEY:-}"  # set this in your env or ~/.env
export FASTVIDEO_PROMPT_CEREBRAS_MODEL="gpt-oss-120b"
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/torchinductor"
export TRITON_CACHE_DIR="$HOME/.triton/cache"
export TORCH_CUDA_ARCH_LIST="10.0a"

# Compiler env (needed for flashinfer JIT compilation at server startup)
export CXX="$CONDA_PREFIX/compiler_compat/g++"
export CC="$CONDA_PREFIX/compiler_compat/gcc"
export CUDAHOSTCXX="$GXX"
export NVCC_PREPEND_FLAGS="-ccbin $GXX -allow-unsupported-compiler"

# Link against libcudart.so.12 at JIT compile time; stubs for libcuda.so
# cuda-compat has libcudart.so -> libcudart.so.12 (linker needs unversioned name)
export LIBRARY_PATH="$CONDA_PREFIX/lib/cuda-compat:$CONDA_PREFIX/lib/stubs"

# Only libcudart.so.12 at runtime — prevents cuDNN from seeing .so.13
export LD_LIBRARY_PATH="$CUDA_RT_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export FASTVIDEO_GPU_COUNT="${FASTVIDEO_GPU_COUNT:-all}"
export DREAMVERSE_SP_SIZE="${DREAMVERSE_SP_SIZE:-4}"
PORT="${DREAMVERSE_PORT:-8009}"

FFMPEG_ENV="$(dirname "$0")/ffmpeg-env.sh"
# shellcheck source=ffmpeg-env.sh
[[ -f "$FFMPEG_ENV" ]] && source "$FFMPEG_ENV"

echo "==> Launching dreamverse-server on GPU $CUDA_VISIBLE_DEVICES port $PORT"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    "$CONDA_PREFIX/bin/dreamverse-server" --host 0.0.0.0 --port "$PORT"
