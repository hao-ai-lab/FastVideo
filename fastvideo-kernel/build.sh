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
if [ "${1:-}" = "--release" ] || [ "${1:-}" = "-r" ]; then
    RELEASE=1
fi

if [ "$RELEASE" -eq 1 ]; then
    # Force-enable ThunderKittens kernels and compile for Hopper.
    # Intended for producing release wheels/images on machines without a GPU.
    export TORCH_CUDA_ARCH_LIST="9.0a"
    export CMAKE_ARGS="${CMAKE_ARGS:-} -DFASTVIDEO_KERNEL_BUILD_TK=ON -DCMAKE_CUDA_ARCHITECTURES=90a"
fi

echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST:-<unset>}"
echo "CMAKE_ARGS: ${CMAKE_ARGS:-<unset>}"
# Build and install
# Use -v for verbose output
pip install . -v --no-build-isolation
