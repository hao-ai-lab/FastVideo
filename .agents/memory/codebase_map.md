# FastVideo-WorldModel — Codebase Map

High-level structural index for agent orientation. Updated 2026-03-02.

## Repository Layout

```
FastVideo-WorldModel/
├── fastvideo/                 # Core Python package
│   ├── models/                # Model implementations
│   │   ├── dits/              #   DiT transformers (wanvideo, ltx2, ...)
│   │   ├── vaes/              #   VAE models
│   │   ├── encoders/          #   Text/image encoders (T5, CLIP)
│   │   ├── schedulers/        #   Noise schedulers
│   │   ├── upsamplers/        #   Super-resolution models
│   │   ├── audio/             #   Audio models
│   │   └── loader/            #   Component loaders for HF repos
│   ├── configs/               # Configuration system
│   │   ├── models/            #   Arch configs + param_names_mapping
│   │   ├── pipelines/         #   Pipeline wiring
│   │   └── sample/            #   Default sampling parameters
│   ├── pipelines/             # End-to-end pipelines
│   │   ├── basic/             #   Per-model pipelines (wan/, ltx2/, ...)
│   │   └── stages/            #   Reusable pipeline stages
│   ├── training/              # Training infrastructure
│   │   ├── trackers.py        #   W&B tracker (BaseTracker → WandbTracker)
│   │   ├── training_utils.py  #   Checkpointing, grad clipping, state dicts
│   │   ├── training_pipeline.py        # Base training pipeline
│   │   ├── wan_training_pipeline.py    # Wan T2V training
│   │   ├── wan_i2v_training_pipeline.py # Wan I2V training
│   │   ├── distillation_pipeline.py    # Distillation base
│   │   ├── wan_distillation_pipeline.py # Wan distillation
│   │   ├── self_forcing_distillation_pipeline.py # Self-forcing distill
│   │   ├── ltx2_training_pipeline.py   # LTX-2 training
│   │   └── matrixgame_training_pipeline.py # MatrixGame training
│   ├── attention/             # Attention backends
│   ├── distributed/           # Sequence/tensor parallel utilities
│   ├── layers/                # Tensor-parallel layers
│   ├── tests/                 # Package-level tests
│   │   ├── training/          #   Training regression tests (W&B summary comparison)
│   │   ├── ssim/              #   SSIM visual regression tests
│   │   ├── encoders/          #   Encoder parity tests
│   │   └── modal/             #   Modal CI test runner
│   └── registry.py            # Unified config registry
├── fastvideo-kernel/          # CUDA/custom kernels (separate build: ./build.sh)
├── scripts/                   # Utility scripts
│   ├── distill/               #   Distillation launch scripts
│   ├── inference/             #   Inference scripts
│   ├── checkpoint_conversion/ #   Weight conversion tools
│   ├── finetune/              #   Finetune scripts
│   └── preprocess/            #   Data preprocessing
├── examples/                  # Ready-to-run examples
│   ├── training/              #   Training examples (finetune/, consistency_finetune/)
│   ├── distill/               #   Distillation examples
│   ├── inference/             #   Inference examples
│   └── dataset/               #   Dataset examples
├── docs/                      # MkDocs documentation source
│   ├── design/overview.md     #   Architecture overview
│   ├── training/              #   Training guides
│   └── contributing/          #   Contributor guides + coding_agents.md
├── tests/                     # Top-level tests (local_tests/)
├── AGENTS.md                  # Agent coding guidelines
└── .agents/                    # Agent infrastructure (you are here)
```

## Key Training Entrypoints

| Pipeline | Entrypoint | Launch Pattern |
|----------|-----------|----------------|
| Wan T2V finetune | `fastvideo/training/wan_training_pipeline.py` | `torchrun --nproc_per_node N` |
| Wan I2V finetune | `fastvideo/training/wan_i2v_training_pipeline.py` | `torchrun --nproc_per_node N` |
| Wan distillation (DMD) | `fastvideo/training/wan_distillation_pipeline.py` | `torchrun --nproc_per_node N` |
| Self-forcing distill | `fastvideo/training/wan_self_forcing_distillation_pipeline.py` | `torchrun --nproc_per_node N` |
| LTX-2 finetune | `fastvideo/training/ltx2_training_pipeline.py` | `torchrun --nproc_per_node N` |
| MatrixGame | `fastvideo/training/matrixgame_training_pipeline.py` | `torchrun --nproc_per_node N` |

## W&B Integration

- **Tracker classes**: `fastvideo/training/trackers.py`
  - `WandbTracker` — logs metrics, videos, timing
  - `SequentialTracker` — fan-out to multiple trackers
  - `DummyTracker` — no-op for offline/test
- **Run summary location**: `<output_dir>/tracker/wandb/latest-run/files/wandb-summary.json`
- **Reference summaries**: `fastvideo/tests/training/*/` (e.g., `a40_reference_wandb_summary.json`)
- **Environment**: `WANDB_API_KEY`, `WANDB_BASE_URL`, `WANDB_MODE`

## Critical Environment Variables

| Variable | Purpose |
|----------|---------|
| `WANDB_API_KEY` | W&B authentication |
| `WANDB_MODE` | `online` / `offline` |
| `FASTVIDEO_ATTENTION_BACKEND` | `FLASH_ATTN` / `TORCH_SDPA` |
| `TOKENIZERS_PARALLELISM` | Set `false` to avoid fork warnings |
| `HF_HOME` | HuggingFace cache directory |

## Build & Test Commands

```bash
uv pip install -e .[dev]                          # Editable install
pre-commit run --all-files                        # Lint/format/spell
pytest tests/                                     # Top-level tests
pytest fastvideo/tests/ -v                        # Package tests
pytest fastvideo/tests/training/Vanilla -srP      # Training loss regression
pytest fastvideo/tests/ssim/ -vs                  # SSIM visual regression
cd fastvideo-kernel && ./build.sh                 # Build kernels
```
