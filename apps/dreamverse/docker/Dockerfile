# syntax=docker/dockerfile:1.7
# CUDA base. CUDA_VERSION/UBUNTU_VERSION feed the default tag (matches the
# unified docker/Dockerfile); override BUILD_BASE_IMAGE wholesale for a mirror.
ARG CUDA_VERSION=13.0.0
ARG UBUNTU_VERSION=22.04
ARG BUILD_BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}
FROM ${BUILD_BASE_IMAGE}

ARG BUILD_FASTVIDEO_KERNEL_FROM_SOURCE=0
ARG BUILD_DREAMVERSE_UI=0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/opt/uv/cache

# pyproject no longer pins a PyTorch index; GPU-less build host -> pin explicitly
# (auto would fall back to CPU). Matches the base CUDA: cu130 (13.0) / cu126 (12.6).
ARG UV_TORCH_BACKEND=cu130
ENV UV_TORCH_BACKEND=${UV_TORCH_BACKEND}

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc-11 g++-11 clang-11 \
        make cmake ninja-build pkg-config nasm \
        git curl wget ca-certificates \
        libssl-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-11

# Version-agnostic /usr/local/cuda symlink so this works for any CUDA_VERSION.
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/root/.local/bin:/opt/venv/bin:${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV VIRTUAL_ENV=/opt/venv

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN uv venv --python 3.12 --seed /opt/venv \
    && echo 'source /opt/venv/bin/activate' >> /root/.bashrc

WORKDIR /opt/FastVideo

COPY . /opt/FastVideo

RUN --mount=type=cache,target=/opt/uv/cache \
    source /opt/venv/bin/activate \
    && uv pip install "/opt/FastVideo[dreamverse]"

# Standard docker build does not expose GPUs, while fastvideo-kernel/build.sh
# detects the CUDA architecture with torch at build time. The FastVideo package
# install above brings in the pinned fastvideo-kernel package; rebuild from the
# copied source only on hosts configured for build-time GPU access.
RUN --mount=type=cache,target=/opt/uv/cache \
    if [[ "${BUILD_FASTVIDEO_KERNEL_FROM_SOURCE}" == "1" ]]; then \
        sed -i 's/^git submodule update --init --recursive$/if [[ -d ..\/.git ]]; then git submodule update --init --recursive; fi/' \
            /opt/FastVideo/fastvideo-kernel/build.sh \
        && source /opt/venv/bin/activate \
        && cd /opt/FastVideo/fastvideo-kernel \
        && ./build.sh; \
    else \
        echo "Skipping source fastvideo-kernel build; using installed fastvideo-kernel package."; \
    fi

RUN if [[ "${BUILD_DREAMVERSE_UI}" == "1" ]]; then \
        curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
        && apt-get install -y --no-install-recommends nodejs \
        && rm -rf /var/lib/apt/lists/* \
        && cd /opt/FastVideo/apps/dreamverse/web \
        && npm ci --ignore-scripts \
        && NEXT_OUTPUT_EXPORT=1 npm run build \
        && rm -rf node_modules .next; \
    else \
        echo "Skipping Dreamverse UI build; Building backend-only image."; \
    fi

# The monorepo ffmpeg installer force-selects conda compiler triplets for
# local dev shells. Inside this image we explicitly opt into the system
# gcc/g++ toolchain.
RUN source /opt/venv/bin/activate \
    && INSTALL_PREFIX=/opt/ffmpeg-native \
       SOURCE_DIR=/tmp/ffmpeg-native-src \
       FFMPEG_NATIVE_CC=/usr/bin/gcc \
       FFMPEG_NATIVE_CXX=/usr/bin/g++ \
       bash /opt/FastVideo/apps/dreamverse/scripts/install_native_ffmpeg.sh

ENV FASTVIDEO_DREAMVERSE_HOME=/var/lib/dreamverse \
    STREAM_MODE=av_fmp4 \
    FASTVIDEO_ENABLE_PROMPT_SAFETY=0 \
    HF_HOME=/root/.cache/huggingface

RUN mkdir -p /var/lib/dreamverse

EXPOSE 8009

HEALTHCHECK --interval=30s --timeout=5s --start-period=2700s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8009/healthz || exit 1

ENTRYPOINT ["/opt/FastVideo/apps/dreamverse/docker/docker_entrypoint.sh"]
CMD ["dreamverse-server", "--host", "0.0.0.0", "--port", "8009"]
