(docker)=
# 🐳 Using the FastVideo Docker Image

If you prefer a containerized development environment or want to avoid managing dependencies manually, you can use our prebuilt Docker image:

**Images:** [`ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:py3.12-latest`](https://ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev)

## Starting the container

```bash
docker run --gpus all -it ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:py3.12-latest
```

This will:

- Start the container with GPU access  
- Drop you into a shell with the `fastvideo-dev` Conda environment preconfigured

## Using the container

```bash
# Conda environment should already be active
# FastVideo package installed in editable mode

# Pull the latest changes from remote
cd /FastVideo
git pull

# Run linters and tests
pre-commit run --all-files
pytest tests/
```
