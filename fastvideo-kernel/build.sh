#!/bin/bash
set -ex

# Simple build script wrapping uv/pip

echo "Building fastvideo-kernel..."

# Ensure submodules are initialized if needed (tk)
# git submodule update --init --recursive

# Install build dependencies
pip install scikit-build-core cmake ninja

# Force TORCH_CUDA_ARCH_LIST to 9.0a to avoid compiling for unsupported architectures (like sm_80)
export TORCH_CUDA_ARCH_LIST="9.0a"

# Build and install
# Use -v for verbose output
pip install . -v --no-build-isolation
