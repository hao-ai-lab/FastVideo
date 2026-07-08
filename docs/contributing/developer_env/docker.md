
# 🐳 Using the FastVideo Docker Image

If you prefer a containerized development environment or want to avoid managing dependencies manually, you can use our prebuilt Docker image:

**Images:** [`ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:py3.12-latest`](https://ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev)

The published FastVideo tags are multi-platform Linux images for `amd64` and
`arm64`; Docker automatically pulls the matching architecture. The
`py3.12-latest` and global `latest` tags select CUDA 12.6.3 with the `cu126`
PyTorch backend. `py3.12-cuda12.6.3-latest` is the explicit alias for that
same image; use `py3.12-cuda13.0.0-latest` for CUDA 13 with `cu130`.

On ARM64, the CUDA 13 image targets DGX Spark (`sm_121`), while the CUDA 12.6
image targets GH200-class hardware (`sm_90a`). These published-tag defaults
are distinct from an unparameterized local `docker build`: `docker/Dockerfile`
itself still defaults to CUDA 13 and `cu130`.

DGX Spark users must therefore select `py3.12-cuda13.0.0-latest`; the default
CUDA 12.6 ARM64 image does not contain an `sm_121` FastVideo kernel.

## Starting the container

```bash
docker run --gpus all -it ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:py3.12-latest
```

This will:

- Start the container with GPU access  
- Drop you into a shell with the FastVideo virtual environment activated

## Building locally

The same Dockerfile builds both architectures and defaults to CUDA 13. On a
native ARM64 host, build the CUDA 13 DGX Spark image with:

```bash
docker build --platform linux/arm64 -f docker/Dockerfile \
  --build-arg TORCH_CUDA_ARCH_LIST=12.1 \
  -t fastvideo-dev:spark .
```

This uses the prebuilt Linux ARM64 FlashAttention wheel. FastVideo's in-tree
CUDA kernel is still compiled for `sm_121` as part of the image build.

## Using the container

```bash
# The FastVideo virtual environment should already be active
# FastVideo package installed in editable mode

# Pull the latest changes from remote
cd /FastVideo
git pull

# Run linters and tests
pre-commit run --all-files
pytest tests/
```
