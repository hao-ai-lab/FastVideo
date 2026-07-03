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
#   # DGX Spark / GB10 (aarch64, Blackwell sm_121): the matching arm64
#   # prebuilt flash-attn wheel is selected automatically from TARGETARCH.
#   docker build --platform linux/arm64 -f docker/Dockerfile \
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

# Cap CMake/Ninja concurrency to fit standard 16 GB build runners. This default
# matches the CUDA 13 kernel-release build; the CUDA 12.6 image workflow uses 4.
ARG CMAKE_BUILD_PARALLEL_LEVEL=1

# flash-attn: when FLASH_ATTN_WHEEL_TAG is set, install the matching prebuilt wheel
# for TARGETARCH. The prebuilt-wheel project publishes x86_64 and arm64 wheels in
# different releases, so each architecture has its own release URL. Set the wheel
# tag to empty only when an explicit source build is desired. The wheel's
# cuXXXtorchY.Y tag must match UV_TORCH_BACKEND (below): cu130torch2.12 for the
# default 13.0 base, cu126torch2.12 for 12.6 (both backends are on torch 2.12).
ARG FLASH_ATTN_VERSION=2.8.3
ARG FLASH_ATTN_WHEEL_TAG=cu130torch2.12
ARG FLASH_ATTN_WHEEL_RELEASE=https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.17
ARG FLASH_ATTN_WHEEL_RELEASE_ARM64=https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.22

# The prebuilt flash-attn wheel ships an FA4 `flash_attn.cute` built against the
# cutlass-4.4 `cute.core.ThrMma` API, which crashes on the cutlass-dsl 4.5 that
# flashinfer/quack pull in. After the wheel install we overlay this cutlass-4.5-safe
# upstream cute (flash-attn-4) so the image runs FA4 instead of the FA2 fallback.
ARG FA4_CUTE_REF=940cd9680f3315f2f06b43ab5bea2c2cf2d96806

# Provided automatically by BuildKit/buildx (e.g. "amd64" / "arm64") and used to
# select the prebuilt flash-attn wheel. Empty under a plain `docker build` without
# BuildKit, which we treat as x86_64 for compatibility with the historical build.
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
# explicitly, matching the base CUDA: cu130 (default, 13.0) / cu126 (12.6), both
# on torch 2.12. Keep FLASH_ATTN_WHEEL_TAG in sync.
ARG UV_TORCH_BACKEND=cu130
ENV UV_TORCH_BACKEND=${UV_TORCH_BACKEND}

# Install uv and source its environment
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    echo 'source $HOME/.local/bin/env' >> /root/.bashrc

# Copy only dependency metadata first to leverage Docker cache. The kernel is
# deliberately excluded from this layer: arm64 resolves it to the in-tree source,
# which needs the complete checkout and CUDA architecture settings below.
COPY pyproject.toml ./
COPY docker/uv-excludes ./docker/uv-excludes

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
    uv pip install --excludes docker/uv-excludes ".[dev]" && \
    PYTAG=cp$(echo "${PYTHON_VERSION}" | tr -d .) && \
    case "${TARGETARCH:-amd64}" in \
        amd64) \
            FLASH_ATTN_WHEEL_PLATFORM=linux_x86_64; \
            FLASH_ATTN_TARGET_RELEASE=${FLASH_ATTN_WHEEL_RELEASE} \
            ;; \
        arm64) \
            FLASH_ATTN_WHEEL_PLATFORM=linux_aarch64; \
            FLASH_ATTN_TARGET_RELEASE=${FLASH_ATTN_WHEEL_RELEASE_ARM64} \
            ;; \
        *) \
            echo "Unsupported TARGETARCH for flash-attn: ${TARGETARCH}" >&2; \
            exit 1 \
            ;; \
    esac && \
    if [ -n "${FLASH_ATTN_WHEEL_TAG}" ]; then \
        echo "Installing prebuilt flash-attn wheel (${FLASH_ATTN_WHEEL_TAG}, ${PYTAG}, ${FLASH_ATTN_WHEEL_PLATFORM})" && \
        uv pip install "${FLASH_ATTN_TARGET_RELEASE}/flash_attn-${FLASH_ATTN_VERSION}+${FLASH_ATTN_WHEEL_TAG}-${PYTAG}-${PYTAG}-${FLASH_ATTN_WHEEL_PLATFORM}.whl"; \
    else \
        echo "Building flash-attn ${FLASH_ATTN_VERSION} from source" && \
        uv pip install flash-attn==${FLASH_ATTN_VERSION} --no-build-isolation; \
    fi

# Overlay the cutlass-4.5-safe upstream FA4 cute (FA4_CUTE_REF) over the
# wheel/source one so the image runs FA4, not the FA2 fallback. This pulls the FA4
# runtime stack (cutlass-dsl, quack-kernels, apache-tvm-ffi, torch-c-dlpack-ext) --
# the same deps the [dreamverse] extra already installs in CI; the installed torch
# is left untouched. The cute package only ships flash_attn/cute (never
# flash_attn/__init__.py), so FA2/varlen/bert_padding stay from the install above;
# rmtree clears the wheel's stale cute files first to avoid an install conflict.
# Then verify both survive so a broken overlay fails the build instead of shipping
# an FA2-less image. x86 only: the FA4 stack (quack-kernels etc.) is unvalidated on
# arm64 / GB10 (sm_121), so there we skip the overlay; FA4 is opt-in
# (FASTVIDEO_FA4=1) and errors if set without the overlay, so leave it unset on
# arm64 and the image runs FA3/FA2 as usual.
RUN --mount=type=cache,target=/opt/uv/cache \
    source $HOME/.local/bin/env && \
    source /opt/venv/bin/activate && \
    if [ "${TARGETARCH}" = "arm64" ]; then \
        echo "Skipping FA4 cute overlay on arm64 (FA4 stack unvalidated there; do not set FASTVIDEO_FA4)"; \
    else \
        python -c "import glob, shutil; [shutil.rmtree(d, ignore_errors=True) for d in glob.glob('/opt/venv/lib/python*/site-packages/flash_attn/cute')]" && \
        uv pip install "flash-attn-4 @ git+https://github.com/Dao-AILab/flash-attention.git@${FA4_CUTE_REF}#subdirectory=flash_attn/cute" && \
        python -c "import flash_attn; assert hasattr(flash_attn, 'flash_attn_func'), 'FA2 was clobbered by the cute overlay'; import flash_attn.cute; print('FA2 + FA4 cute OK')"; \
    fi

COPY . .

# Install FastVideo Unified Kernel exactly once. build.sh initializes only its
# CUTLASS/ThunderKittens submodules, then compiles for TORCH_CUDA_ARCH_LIST
# (default Hopper sm_90a) without requiring a live GPU.
RUN --mount=type=cache,target=/opt/uv/cache \
    source $HOME/.local/bin/env && \
    source /opt/venv/bin/activate && \
    cd fastvideo-kernel && \
    CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} ./build.sh

# Install FastVideo itself (editable) now that the source is present, and set up
# shell configuration. Dependencies and the local kernel are already installed,
# so --no-deps prevents uv from resolving/building the ARM source override again.
RUN --mount=type=cache,target=/opt/uv/cache \
    source $HOME/.local/bin/env && \
    source /opt/venv/bin/activate && \
    uv pip install -e ".[dev]" --no-deps && \
    (git config --unset-all http.https://github.com/.extraheader || true) && \
    echo 'source /opt/venv/bin/activate' >> /root/.bashrc && \
    echo 'if [ -n "$ZSH_VERSION" ] && [ -f ~/.zshrc ]; then . ~/.zshrc; elif [ -f ~/.bashrc ]; then . ~/.bashrc; fi' > /root/.profile

EXPOSE 22
