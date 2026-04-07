
# 🛠️ Contributing to FastVideo

Thank you for your interest in contributing to FastVideo. We want the process
to be smooth and beginner‑friendly, whether you are adding a new pipeline,
improving performance, or fixing a bug.

## Quick prerequisites

- **OS**: Linux is the primary development target (WSL can work).
- **GPU**: NVIDIA GPU recommended for inference and training workflows.
- **CUDA**: Use a recent CUDA 12.x toolchain (see the installation guide for
  the current recommendation).

For a full install checklist, see `docs/getting_started/installation/gpu.md`.

## Local development (UV + editable install)

If you previously used Conda for local setup, switch to uv for a faster and more stable development environment.

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
wget -qO- https://astral.sh/uv/install.sh | sh
```

Create and activate a uv environment (recommended):

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
```

Conda alternative (supported):

```bash
conda create -n fastvideo python=3.12 -y
conda activate fastvideo
```

Clone the repo:

```bash
git clone https://github.com/hao-ai-lab/FastVideo.git && cd FastVideo
```

Install FastVideo in editable mode and set up hooks:

```bash
uv pip install -e .[dev]

# Optional: FlashAttention (builds native kernels)
uv pip install flash-attn --no-build-isolation -v

# Linting, formatting, static typing
pre-commit install --hook-type pre-commit --hook-type commit-msg
pre-commit run --all-files

# Unit tests
pytest tests/
```

If you are on a Hopper GPU, installing FlashAttention 3 can improve
performance (see `docs/inference/optimizations.md`).

## Docker development (optional)

If you prefer a containerized environment, use the dev image documented in
`docs/contributing/developer_env/docker.md`.

## Testing

See the [Testing Guide](testing.md) for how to add and run tests in FastVideo.

## Attention backend development

If you are adding a new attention kernel or backend, follow
[Attention Backend Development](attention_backend.md).

## Contributing with coding agents

For a step‑by‑step workflow on adding pipelines or components with coding
agents, see `docs/contributing/coding_agents.md`.
