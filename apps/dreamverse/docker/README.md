# Dreamverse Docker Image

This folder contains the Docker image for Dreamverse inside the FastVideo
monorepo. Build commands use the FastVideo repository root as the Docker
context, so run the helper scripts from this folder or from any path in the
checkout.

## Build

```bash
apps/dreamverse/docker/docker_build.sh
```

The image defaults to `dreamverse:dev`. Override it with:

```bash
DREAMVERSE_IMAGE=dreamverse:local apps/dreamverse/docker/docker_build.sh
```

Backend-only remains the default image. To include the static Dreamverse UI
served by the backend, set `BUILD_DREAMVERSE_UI=1` and choose a specific image
tag:

```bash
BUILD_DREAMVERSE_UI=1 DREAMVERSE_IMAGE=<image-tag> apps/dreamverse/docker/docker_build.sh
```

Prefer SHA-specific tags for deployable images; avoid `latest`.

The Dockerfile defaults to CUDA 13.0.0 with the cu130 PyTorch backend. CI also
builds a CUDA 12.6.3 / cu126 image. Select that local build explicitly with:

```bash
CUDA_VERSION=12.6.3 apps/dreamverse/docker/docker_build.sh
```

The helper derives `UV_TORCH_BACKEND=cu126` for CUDA 12.x and `cu130` for CUDA
13.x. Set `UV_TORCH_BACKEND` explicitly for a custom CUDA version. The legacy
complete-image-tag override remains supported and selects the same matching
backend:

```bash
CUDA_TAG=12.6.3-cudnn-devel-ubuntu22.04 apps/dreamverse/docker/docker_build.sh
```

Do not set `CUDA_TAG` and `CUDA_VERSION` together. The image installs FastVideo
from this checkout with the `dreamverse` extra, including the FA4
flash-attention fork and FlashInfer for NVFP4 quantization, and builds native
FFmpeg.

FastVideo's pinned `fastvideo-kernel==0.3.1` package is installed by default.
To rebuild `fastvideo-kernel` from this checkout during the image build, set:

```bash
BUILD_FASTVIDEO_KERNEL_FROM_SOURCE=1 apps/dreamverse/docker/docker_build.sh
```

That source build detects the GPU architecture with torch during `docker
build`. On hosts where Docker does not expose GPUs during build, leave the
default package install path enabled.

## Run

```bash
CEREBRAS_API_KEY="<your-key>" \
GROQ_API_KEY="<your-key>" \
apps/dreamverse/docker/docker_run.sh
```

The container serves Dreamverse on host port `8009` by default and mounts:

```text
$HOME/.cache/huggingface -> /root/.cache/huggingface
apps/dreamverse/outputs -> /var/lib/dreamverse/outputs
```

Override the host port and output directory with `BACKEND_PORT` and
`DREAMVERSE_OUTPUTS_DIR`.

To pin the container to a specific host GPU, pass Docker's GPU request syntax:

```bash
DREAMVERSE_DOCKER_GPUS=device=4 FASTVIDEO_GPU_COUNT=1 \
  CEREBRAS_API_KEY="<your-key>" \
  GROQ_API_KEY="<your-key>" \
  apps/dreamverse/docker/docker_run.sh
```

## Smoke

```bash
CEREBRAS_API_KEY=placeholder \
GROQ_API_KEY=placeholder \
apps/dreamverse/docker/docker_smoke.sh
```

The smoke script starts the container, polls `/healthz`, then polls `/readyz`.
It removes the container on exit unless `DREAMVERSE_KEEP_CONTAINER=1` is set.
