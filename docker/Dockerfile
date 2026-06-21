# syntax=docker/dockerfile:1.7
# =============================================================================
# FastVideo CUDA development image (unified, parameterized)
# =============================================================================
# Replaces the former per-Python Dockerfiles
# (Dockerfile.python3.10 / 3.11 / 3.12 / 3.12.cuda12.9.1): pick a variant with
# --build-arg instead of maintaining a separate near-identical file.
#
#   # Default image (Python 3.12, CUDA 13.0.0, Ubuntu 22.04, Hopper sm_90a):
#   docker build -f docker/Dockerfile -t fastvideo-dev:py3.12 .
#
#   # Python 3.10 / 3.11:
#   docker build -f docker/Dockerfile --build-arg PYTHON_VERSION=3.11 \
#       -t fastvideo-dev:py3.11 .
#
#   # CUDA 13.0.0: match the torch backend + flash-attn wheel to the base CUDA.
#   docker build -f docker/Dockerfile \
#       --build-arg CUDA_VERSION=13.0.0 \
#       --build-arg UV_TORCH_BACKEND=cu130 \
#       --build-arg FLASH_ATTN_WHEEL_TAG=cu130torch2.12 \
#       -t fastvideo-dev:py3.12-cuda13.0.0 .
#
#   # DGX Spark / GB10 (aarch64, Blackwell sm_121): build on an arm64 host;
#   # the x86-only prebuilt flash-attn wheel is skipped automatically.
#   docker build -f docker/Dockerfile \
#       --build-arg TORCH_CUDA_ARCH_LIST=12.1 -t fastvideo-dev:spark .
#
# The ARG defaults below are the source of truth for the default image.
# ROCm builds use docker/rocm7_1.Dockerfile (different base + flow).
# =============================================================================

# --- Base image selection (overridable for private mirrors / new CUDA) -------
# Build on the -devel- image: nvcc + the full toolkit are needed to compile
# fastvideo-kernel. CUDA_VERSION/UBUNTU_VERSION feed the default tag, or override
# BUILD_BASE_IMAGE wholesale for a mirror.
ARG CUDA_VERSION=13.0.0
ARG UBUNTU_VERSION=22.04
ARG BUILD_BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}

FROM ${BUILD_BASE_IMAGE} AS base

# Redeclare the ARGs we need inside the stage (FROM consumes the global ones).
ARG CUDA_VERSION
ARG PYTHON_VERSION=3.12

# CUDA arch fastvideo-kernel is compiled for. Build machines have no GPU, so
# build.sh uses this instead of probing a live device. Default Hopper (sm_90a),
# matching the released kernel wheel; override e.g. TORCH_CUDA_ARCH_LIST=12.1
# for GB10/Blackwell (DGX Spark).
ARG TORCH_CUDA_ARCH_LIST=9.0a

# flash-attn: when FLASH_ATTN_WHEEL_TAG is set (and the target is x86_64), install
# the matching prebuilt wheel; otherwise -- empty tag, or an arm64 target where no
# prebuilt wheel exists -- build it from source. The wheel's cuXXXtorchY.Y tag must
# match UV_TORCH_BACKEND (below): cu130torch2.12 for the default 13.0 base,
# cu126torch2.12 for 12.6, cu128torch2.11 for 12.8 (cu126/cu130 -> torch 2.12,
# cu128 -> torch 2.11).
ARG FLASH_ATTN_VERSION=2.8.3
ARG FLASH_ATTN_WHEEL_TAG=cu130torch2.12
ARG FLASH_ATTN_WHEEL_RELEASE=https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.17

# Provided automatically by buildx (e.g. "amd64" / "arm64"); used to skip the
# x86-only prebuilt flash-attn wheel on arm64. Empty under a plain `docker build`
# without buildx, which we treat as x86_64 (the historical default).
ARG TARGETARCH

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

WORKDIR /FastVideo

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    ca-certificates \
    openssh-server \
    zsh \
    vim \
    curl \
    gcc-11 \
    g++-11 \
    clang-11 \
    && rm -rf /var/lib/apt/lists/*

# Set up C++20 compilers for ThunderKittens
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 --slave /usr/bin/g++ g++ /usr/bin/g++-11

# Set CUDA environment variables. Use the version-agnostic /usr/local/cuda
# symlink the nvidia images provide so this works for any CUDA_VERSION.
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

# The uv package cache lives in a BuildKit cache mount (see RUNs below), not in an
# image layer, so we can drop the old --no-cache-dir to speed up rebuilds without
# bloating the image. Copy link mode avoids hardlink errors across the mount.
ENV UV_CACHE_DIR=/opt/uv/cache
ENV UV_LINK_MODE=copy

# pyproject no longer pins a PyTorch index; select the build at resolve time.
# Build hosts have no GPU driver (auto-detect would fall back to CPU), so pin
# explicitly, matching the base CUDA: cu130 (default, 13.0) / cu126 (12.6) ->
# torch 2.12, cu128 (12.8) -> torch 2.11. Keep FLASH_ATTN_WHEEL_TAG in sync.
ARG UV_TORCH_BACKEND=cu130
ENV UV_TORCH_BACKEND=${UV_TORCH_BACKEND}

# Install uv and source its environment
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    echo 'source $HOME/.local/bin/env' >> /root/.bashrc

# Copy just the pyproject.toml first to leverage Docker cache
COPY pyproject.toml ./

# Create a dummy README to satisfy the installation
RUN echo "# Placeholder" > README.md

# Create the venv and install the (slow-changing) dependency tree + flash-attn
# from pyproject.toml alone, before the source is copied in, so that editing
# source below does not bust this expensive layer.
RUN --mount=type=cache,target=/opt/uv/cache \
    source $HOME/.local/bin/env && \
    uv venv --python ${PYTHON_VERSION} --seed /opt/venv && \
    source /opt/venv/bin/activate && \
    uv pip install --upgrade pip && \
    uv pip install ".[dev]" && \
    PYTAG=cp$(echo "${PYTHON_VERSION}" | tr -d .) && \
    if [ -n "${FLASH_ATTN_WHEEL_TAG}" ] && [ "${TARGETARCH}" != "arm64" ]; then \
        echo "Installing prebuilt flash-attn wheel (${FLASH_ATTN_WHEEL_TAG}, ${PYTAG})" && \
        uv pip install "${FLASH_ATTN_WHEEL_RELEASE}/flash_attn-${FLASH_ATTN_VERSION}+${FLASH_ATTN_WHEEL_TAG}-${PYTAG}-${PYTAG}-linux_x86_64.whl"; \
    else \
        echo "Building flash-attn ${FLASH_ATTN_VERSION} from source" && \
        uv pip install flash-attn==${FLASH_ATTN_VERSION} --no-build-isolation; \
    fi

COPY . .

# Install FastVideo itself (editable) now that the source is present, and set up
# shell configuration. Most dependencies are already satisfied by the layer above.
RUN --mount=type=cache,target=/opt/uv/cache \
    source $HOME/.local/bin/env && \
    source /opt/venv/bin/activate && \
    uv pip install -e ".[dev]" && \
    git config --unset-all http.https://github.com/.extraheader || true && \
    echo 'source /opt/venv/bin/activate' >> /root/.bashrc && \
    echo 'if [ -n "$ZSH_VERSION" ] && [ -f ~/.zshrc ]; then . ~/.zshrc; elif [ -f ~/.bashrc ]; then . ~/.bashrc; fi' > /root/.profile

# Install FastVideo Unified Kernel. The arch comes from TORCH_CUDA_ARCH_LIST
# (default Hopper sm_90a) so build.sh does not need a live GPU to probe.
RUN --mount=type=cache,target=/opt/uv/cache \
    source $HOME/.local/bin/env && \
    source /opt/venv/bin/activate && \
    cd fastvideo-kernel && \
    git submodule update --init --recursive && \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} ./build.sh

EXPOSE 22
