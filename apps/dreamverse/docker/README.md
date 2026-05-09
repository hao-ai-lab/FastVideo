# Dreamverse Docker Image

This folder contains the backend-only Docker image for Dreamverse inside the
FastVideo monorepo. Build commands use the FastVideo repository root as the
Docker context, so run the helper scripts from this folder or from any path in
the checkout.

## Build

```bash
apps/dreamverse/docker/docker_build.sh
```

The image defaults to `dreamverse:dev`. Override it with:

```bash
DREAMVERSE_IMAGE=dreamverse:local apps/dreamverse/docker/docker_build.sh
```

The Dockerfile builds a CUDA 12.9.1 image, installs FastVideo from this
checkout with the `dreamverse` extra, installs the FA4
flash-attention fork, builds native FFmpeg, and installs FlashInfer for NVFP4
quantization.

FastVideo's pinned `fastvideo-kernel==0.2.6` package is installed by default.
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
