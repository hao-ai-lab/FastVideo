#!/usr/bin/env bash
# setup-dreamverse-env.sh — create and configure the dreamverse conda env
# from scratch on this aarch64 NFS Slurm cluster.
#
# Run from the login node (from the repo root):
#   bash apps/dreamverse/scripts/setup-dreamverse-env.sh
#
# After this script completes, use launch-dreamverse.sh on a compute node.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
ENV_NAME="dreamverse"
LOCAL_DIR="/mnt/local/hal-kevin"       # cache/pkgs — keep on local disk
CONDA_PREFIX="$HOME/miniconda3/envs/$ENV_NAME"  # env — on shared NFS so it survives node changes

echo "==> Removing existing env if present"
conda env remove -p "$CONDA_PREFIX" -y 2>/dev/null || true
rm -rf "$CONDA_PREFIX" 2>/dev/null || true

echo "==> Creating conda env at $CONDA_PREFIX"
CONDA_PKGS_DIRS="$LOCAL_DIR/conda/pkgs" conda create -p "$CONDA_PREFIX" python=3.11 -y

GXX="$CONDA_PREFIX/bin/aarch64-conda-linux-gnu-g++"
GCC="$CONDA_PREFIX/bin/aarch64-conda-linux-gnu-gcc"

echo "==> Installing compiler"
CONDA_PKGS_DIRS="$LOCAL_DIR/conda/pkgs" conda install -p "$CONDA_PREFIX" gxx_linux-aarch64 -y

echo "==> Installing CUDA toolkit (nvcc + headers)"
CONDA_PKGS_DIRS="$LOCAL_DIR/conda/pkgs" conda install -p "$CONDA_PREFIX" -c nvidia cuda-toolkit -y

echo "==> Hiding conflicting libcudart.so.13 immediately"
mkdir -p "$CONDA_PREFIX/lib/hidden"
mv "$CONDA_PREFIX"/lib/libcudart.so* "$CONDA_PREFIX/lib/hidden/" 2>/dev/null || true

echo "==> Fixing compiler_compat symlinks"
mkdir -p "$CONDA_PREFIX/compiler_compat"
ln -sf "$GXX" "$CONDA_PREFIX/compiler_compat/g++"
ln -sf "$GCC" "$CONDA_PREFIX/compiler_compat/gcc"

echo "==> Symlinking CUDA headers to standard location"
for f in "$CONDA_PREFIX/targets/sbsa-linux/include/"*; do
    ln -sf "$f" "$CONDA_PREFIX/include/$(basename "$f")" 2>/dev/null || true
done

echo "==> Installing ffmpeg (native build with x264 + NVENC)"
CUDA_PREFIX="$CONDA_PREFIX" bash "$REPO_ROOT/apps/dreamverse/scripts/install_native_ffmpeg.sh"

echo "==> Installing pip and uv"
CONDA_PKGS_DIRS="$LOCAL_DIR/conda/pkgs" conda install -p "$CONDA_PREFIX" pip -y
"$CONDA_PREFIX/bin/pip" install uv

echo "==> Setting compiler env vars"
export UV_CACHE_DIR="$LOCAL_DIR/cache"
export UV_LINK_MODE=copy
export CXX="$CONDA_PREFIX/compiler_compat/g++"
export CC="$CONDA_PREFIX/compiler_compat/gcc"
export CUDAHOSTCXX="$GXX"
export NVCC_PREPEND_FLAGS="-ccbin $GXX -allow-unsupported-compiler"
export CUDA_HOME="$CONDA_PREFIX"
# Only build for GB200 (sm_100a); CUDA 13 dropped support for older archs
export TORCH_CUDA_ARCH_LIST="10.0a"

echo "==> Installing torch with CUDA 12.8"
UV_CACHE_DIR="$LOCAL_DIR/cache" "$CONDA_PREFIX/bin/uv" pip install torch==2.11.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu128

echo "==> Hiding any newly introduced libcudart.so.13"
mv "$CONDA_PREFIX"/lib/libcudart.so* "$CONDA_PREFIX/lib/hidden/" 2>/dev/null || true

# Set paths now that torch (and its nvidia packages) are installed
CUDA_RT_DIR="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib"
CUDA_RT_SO="$(ls "$CUDA_RT_DIR"/libcudart.so.* 2>/dev/null | head -1)"

# The pip nvidia package only has libcudart.so.12 (versioned), not libcudart.so.
# The linker needs the unversioned name to satisfy -lcudart. Create a compat dir.
mkdir -p "$CONDA_PREFIX/lib/cuda-compat"
ln -sf "$CUDA_RT_SO" "$CONDA_PREFIX/lib/cuda-compat/libcudart.so"

export LIBRARY_PATH="$CONDA_PREFIX/lib/cuda-compat:$CONDA_PREFIX/lib/stubs"
export CMAKE_ARGS="-DCUDA_CUDART_LIBRARY=$CUDA_RT_SO -DCUDA_INCLUDE_DIRS=$CONDA_PREFIX/targets/sbsa-linux/include"

echo "==> Installing build tools"
"$CONDA_PREFIX/bin/pip" install scikit-build-core cmake ninja

echo "==> Initializing git submodules"
cd "$REPO_ROOT"
git submodule update --init fastvideo-kernel/include/cutlass fastvideo-kernel/include/tk

echo "==> Building fastvideo-kernel from local source"
UV_CACHE_DIR="$LOCAL_DIR/cache" "$CONDA_PREFIX/bin/uv" pip install \
    -e "./fastvideo-kernel" --no-build-isolation

echo "==> Installing fastvideo + dreamverse extras"
UV_CACHE_DIR="$LOCAL_DIR/cache" "$CONDA_PREFIX/bin/uv" pip install \
    -e ".[dreamverse]" --no-build-isolation

echo "==> Installing flashinfer-python (pinned, must be last)"
UV_CACHE_DIR="$LOCAL_DIR/cache" "$CONDA_PREFIX/bin/uv" pip install \
    https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.11.post3/flashinfer_python-0.6.11.post3-py3-none-any.whl

echo ""
echo "Done. On a compute node run:"
echo "  CUDA_VISIBLE_DEVICES=0 bash apps/dreamverse/scripts/launch-dreamverse.sh"
