# `fastvideo/` — Core Package

Inference + training framework for video DiTs. Public API entry: `from fastvideo import VideoGenerator, PipelineConfig, SamplingParam`.

## Public Surface (`__init__.py`)

```python
VideoGenerator   # entrypoints/video_generator.py — high-level inference handle
PipelineConfig   # configs/pipelines/base.py — pipeline wiring dataclass
SamplingParam    # api/sampling_param.py — runtime sampling knobs
```

CLI entry: `fastvideo` script → `entrypoints/cli/main.py` (subcommands: `generate`, `serve`, `bench`).

## Layout

```
fastvideo/
├── api/             # Schema + presets for the OpenAI-compatible serving layer
├── attention/       # Backends + selector (FlashAttn / SageAttn / SDPA / VSA / VMoBA / SLA)
├── configs/         # Per-model arch configs + per-pipeline configs (registry-driven)
├── dataset/         # Dataloaders (pre-commit excluded — minimal lint surface)
├── distributed/     # SP/TP groups, device communicators, init helpers
├── entrypoints/     # cli/, openai/, streaming/, video_generator.py
├── hooks/           # Runtime hook system for pipelines
├── layers/          # Tensor-parallel linears + attention wrappers (port targets)
├── models/          # DiT / VAE / encoder / scheduler / loader (pre-commit excluded)
├── pipelines/       # basic/<model>/, preprocess/, stages/, training/
├── platforms/       # CUDA/ROCm capability + AttentionBackendEnum
├── third_party/     # Vendored externals (lint excluded; do not reformat)
├── train/           # NEW modular trainer — methods × models × callbacks
├── training/        # LEGACY monolithic *_training/distillation_pipeline.py
├── worker/          # Multi-process / Ray executors
├── workflow/        # Preprocessing workflow base class
├── registry.py      # Pipeline-config + model-class lookup (canonical)
├── envs.py          # Env-var declarations
├── fastvideo_args.py# Runtime arg dataclass passed through pipelines
└── utils.py         # FlexibleArgumentParser, qualname resolver, etc.
```

## Where to Look

| Task | Location |
|------|----------|
| Add a new pipeline class | `pipelines/basic/<model>/` + `configs/pipelines/<model>.py` + register in `registry.py` |
| Add a new model component | `models/<role>/<model>.py` + `configs/models/<role>/<model>.py` |
| Wire an existing model into a new pipeline | `pipelines/basic/<model>/presets.py` + reuse stages from `pipelines/stages/` |
| Add a converter | `scripts/checkpoint_conversion/<model>_to_*.py` (separate dir, separate AGENTS.md) |
| Add an attention backend | `attention/backends/<name>.py` + register in selector |
| Add a runtime CLI flag | `fastvideo_args.py` (avoid `argparse` ad-hoc inside stages) |

## Conventions Specific Here

- `PipelineStage` subclasses (`pipelines/stages/`) own one verb each (encode, schedule, denoise, decode). Compose, don't fork.
- Every pipeline reads from a `PipelineConfig` subclass and a `SamplingParam`. Never read raw env vars inside a stage — go through `fastvideo.envs`.
- Logger setup: `from fastvideo.logger import init_logger; logger = init_logger(__name__)`. Do not call `logging.getLogger` directly.
- Imports between `train/` and `training/` are **forbidden** — they are independent stacks.

## Pre-Commit Exclusions (do not assume linted)

These dirs are listed in `.pre-commit-config.yaml` `exclude`:

- `fastvideo/third_party/`, `fastvideo/dataset/`, `fastvideo/models/`

Editing files there will NOT trigger yapf/ruff/mypy/codespell. Format manually if a sibling file shows clear style; do not introduce new violations.
