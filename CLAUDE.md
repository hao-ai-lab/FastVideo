# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastVideo is a unified post-training and inference framework for accelerated video generation. It supports video diffusion models (Wan, HunyuanVideo, LTX, Cosmos, StepVideo, etc.) with distributed training (FSDP2, sequence parallelism), sparse attention backends (STA, VSA), and distillation (DMD2).

## Build & Development Commands

```bash
# Install (editable, with dev extras)
uv pip install -e .[dev]

# Build CUDA kernel extensions
cd fastvideo-kernel && ./build.sh

# Lint and format (requires: pre-commit install --hook-type pre-commit --hook-type commit-msg)
pre-commit run --all-files

# Tests
pytest tests/                       # top-level tests
pytest fastvideo/tests/ -v          # package tests
pytest fastvideo/tests/ssim/ -vs    # SSIM regression tests (GPU-heavy)

# Run a single test file
pytest fastvideo/tests/path/to/test_file.py -v

# flash-attn must be installed separately:
# pip install flash-attn==2.8.1 --no-cache-dir --no-build-isolation
```

## Architecture

**Entry points:**
- `fastvideo/entrypoints/video_generator.py` — `VideoGenerator`: main user-facing API (`from fastvideo import VideoGenerator`)
- `fastvideo/entrypoints/cli/main.py` — CLI (`fastvideo <command>`)
- `fastvideo/entrypoints/streaming_generator.py` — streaming video generation

**Config system** (typed dataclasses in `fastvideo/configs/`):
- `models/` — architecture configs per model family
- `pipelines/` — pipeline wiring (which components compose a pipeline)
- `sample/` — sampling parameters

**Pipeline composition:**
- `fastvideo/pipelines/composed_pipeline_base.py` — `ComposedPipelineBase`, base for all pipelines
- `fastvideo/pipelines/stages/` — reusable pipeline stages composed together
- `fastvideo/pipelines/basic/` — end-to-end pipelines per model family
- `ForwardBatch` carries prompts, latents, and timesteps across stages

**Models:**
- `fastvideo/models/dits/` — diffusion transformer implementations (Wan, LTX, HunyuanVideo, etc.)
- `fastvideo/models/encoders/` — text/image encoders (T5, CLIP)
- `fastvideo/models/vaes/` — VAE models
- `fastvideo/models/loader/` — HuggingFace Diffusers-compatible component loading

**Distributed & parallelism:**
- `fastvideo/distributed/` — FSDP2, sequence parallel, tensor parallel, device communication
- `fastvideo/layers/` — tensor-parallel layers, LoRA, quantization

**Attention backends** (`fastvideo/attention/`):
- Selectable via `FASTVIDEO_ATTENTION_BACKEND` env var: `VIDEO_SPARSE_ATTN`, `FLASH_ATTN`, `TORCH_SDPA`

**Registry** (`fastvideo/registry.py`): dynamic registration and resolution of models, pipelines, and configs.

**Training** (`fastvideo/training/`): training utilities, distillation, checkpointing. Training pipelines in `fastvideo/pipelines/training/`.

## Code Style

- Python 3.10+, 80-char line length, 4-space indentation
- Formatting: `yapf`. Linting: `ruff`. Type checking: `mypy`. Spell check: `codespell`
- `snake_case` for functions/files, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants

## Commit Style

Use tag prefixes: `[bugfix]: ...`, `[feat]: ...`, `[misc]: ...`, `[perf]: ...`. Include PR reference `(#1234)` when applicable. Keep commits focused by concern.

## Key Environment Variables

- `FASTVIDEO_ATTENTION_BACKEND` — attention backend selection
- `HF_API_KEY` — HuggingFace credentials for model downloads
- `WANDB_API_KEY` — Weights & Biases experiment tracking
