#!/bin/bash
set -ex

# Simple build script wrapping uv/pip

echo "Building fastvideo-kernel..."

# Ensure submodules are initialized if needed (tk)
# git submodule update --init --recursive

# Install build dependencies
pip install scikit-build-core cmake ninja

# Set TORCH_CUDA_ARCH_LIST only if not already set
if [ -z "$TORCH_CUDA_ARCH_LIST" ]; then
    export TORCH_CUDA_ARCH_LIST="9.0a"
fi

# Build and install
# Use -v for verbose output
pip install . -v --no-build-isolation
