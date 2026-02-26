#!/bin/bash
set -ex

# Simple build script wrapping uv/pip
# Usage:
#   ./build.sh                # local dev build (auto-detect / skip TK kernels when not available)
#   ./build.sh --release      # force-enable Hopper/TK kernels for release builds (no GPU required)

echo "Building fastvideo-kernel..."

# Ensure submodules are initialized if needed (tk)
git submodule update --init --recursive

# Install build dependencies
pip install scikit-build-core cmake ninja

RELEASE=0
GPU_BACKEND=CUDA
for arg in "$@"; do
    case "$arg" in
        --rocm)
            GPU_BACKEND=ROCM
            ;;
    esac
done

# Force-enable ThunderKittens kernels and compile for Hopper.
export TORCH_CUDA_ARCH_LIST="9.0a"
export CMAKE_ARGS="${CMAKE_ARGS:-} -DFASTVIDEO_KERNEL_BUILD_TK=ON -DCMAKE_CUDA_ARCHITECTURES=90a"

export CMAKE_ARGS="${CMAKE_ARGS:-} -DGPU_BACKEND=${GPU_BACKEND}"

echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST:-<unset>}"
echo "CMAKE_ARGS: ${CMAKE_ARGS:-<unset>}"
echo "GPU_BACKEND: ${GPU_BACKEND:-<unset>}"
# Build and install
# Use -v for verbose output
pip install . -v --no-build-isolation