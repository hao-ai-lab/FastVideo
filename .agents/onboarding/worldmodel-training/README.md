# WorldModel Training — Agent Onboarding

Specialized onboarding for agents working on FastVideo-WorldModel training,
distillation, and evaluation. Read the master onboarding (`.agents/onboarding/README.md`)
first, then come here.

---

## Domain Context

FastVideo-WorldModel trains **interactive world models** — video generation systems
that respond to user actions (keyboard/mouse) in real-time. The architecture is
based on **Wan2.1** (SkyReels-V2) DiT models with causal attention for
auto-regressive streaming generation.

**Key techniques you will work with:**
- Full finetuning and LoRA on Wan / LTX-2 / MatrixGame models
- DMD-based distillation (few-step generation)
- Self-Forcing distillation (causal streaming)
- Diffusion-Forcing SFT (DFSFT) for causal models
- VSA (Variable Sparsity Acceleration) for efficient training

---

## Training Code: Two Generations

### New modular framework: `fastvideo/train/` (preferred)

The refactored training code uses a **YAML-only config-driven** architecture
with composable methods, per-role models, and a callback system. All new
training work should use this framework.

### Legacy pipelines: `fastvideo/training/` (deprecated)

The old monolithic pipeline classes (`WanTrainingPipeline`,
`DistillationPipeline`, etc.) still exist but are being phased out. The new
framework imports select utilities from `fastvideo/training/` for backward
compatibility (EMA, gradient clipping, checkpoint wrappers).

---

## Essential Reading (Training-Specific)

Read these **in order** before touching any training code:

| # | File | What You Learn |
|---|------|----------------|
| 1 | `docs/training/overview.md` | Training data flow: raw video → text embeddings + video latents → training |
| 2 | `docs/training/finetune.md` | Training arguments, parallelism (SP/TP), LoRA, validation settings |
| 3 | `docs/training/data_preprocess.md` | How to preprocess datasets into the expected format |
| 4 | `docs/design/overview.md` | Architecture: models, pipelines, configs, registry |

---

## New Training Framework (`fastvideo/train/`)

### Architecture Overview

```
fastvideo/train/
├── __init__.py                    → exports Trainer
├── trainer.py                     → main training loop coordinator
├── entrypoint/
│   ├── train.py                   → YAML-only training entrypoint
│   └── dcp_to_diffusers.py        → checkpoint conversion utility
├── methods/                       → training algorithms (TrainingMethod ABC)
│   ├── base.py                    → TrainingMethod base class
│   ├── fine_tuning/
│   │   ├── finetune.py            → FineTuneMethod (supervised finetuning)
│   │   └── dfsft.py               → DiffusionForcingSFTMethod (causal)
│   ├── distribution_matching/
│   │   ├── dmd2.py                → DMD2Method (distribution matching distill)
│   │   └── self_forcing.py        → SelfForcingMethod (causal streaming)
│   ├── knowledge_distillation/    → (stub, not yet implemented)
│   └── consistency_model/         → (stub, not yet implemented)
├── models/                        → per-role model instances
│   ├── base.py                    → ModelBase & CausalModelBase (ABC)
│   └── wan/
│       ├── wan.py                 → WanModel (non-causal)
│       └── wan_causal.py          → WanCausalModel (causal streaming)
├── callbacks/                     → training hooks & monitoring
│   ├── callback.py                → Callback base class + CallbackDict
│   ├── grad_clip.py               → GradNormClipCallback
│   ├── ema.py                     → EMACallback (shadow weights)
│   └── validation.py              → ValidationCallback (sampling + eval)
└── utils/                         → configuration, building, checkpointing
    ├── builder.py                 → build_from_config() (config → runtime)
    ├── checkpoint.py              → CheckpointManager (DCP-based)
    ├── config.py                  → load_run_config() (YAML → RunConfig)
    ├── training_config.py         → TypedConfig dataclasses
    ├── optimizer.py               → build_optimizer_and_scheduler()
    ├── instantiate.py             → resolve_target() + instantiate()
    ├── tracking.py                → build_tracker() (W&B, etc.)
    ├── dataloader.py              → dataloader utilities
    ├── module_state.py            → apply_trainable()
    └── moduleloader.py            → load_module_from_path()
```

### Key Concepts

**TrainingMethod** (`methods/base.py`): Abstract base class for all training
algorithms. Owns role models (student, teacher, critic), manages checkpoint
state, and defines the training step interface.

**ModelBase** (`models/base.py`): Per-role model wrapper. Each role (student,
teacher, critic) gets its own `ModelBase` instance owning a `transformer` and
`noise_scheduler`. `CausalModelBase` extends this for streaming models.

**Callback system** (`callbacks/`): Composable hooks for gradient clipping,
EMA, validation, etc. Configured via YAML, dispatched by `CallbackDict`.

**Config system** (`utils/config.py`, `utils/training_config.py`): YAML files
are parsed into typed `RunConfig` dataclass trees. Models and methods use
`_target_` fields for instantiation (similar to Hydra).

### Training Flow

```
run_training_from_config(config_path)
  → load_run_config()           # YAML → RunConfig
  → init_distributed()          # TP/SP setup
  → build_from_config()         # instantiate models, method, dataloader
  → Trainer.run()               # main loop:
      ├─ callbacks.on_train_start()
      ├─ checkpoint_manager.maybe_resume()
      ├─ for step in range(max_steps):
      │    ├─ method.single_train_step(batch)
      │    ├─ method.backward()
      │    ├─ callbacks.on_before_optimizer_step()
      │    ├─ method.optimizers_schedulers_step()
      │    ├─ tracker.log(metrics, step)
      │    ├─ callbacks.on_training_step_end()
      │    └─ checkpoint_manager.maybe_save(step)
      ├─ callbacks.on_train_end()
      └─ checkpoint_manager.save_final()
```

### Training Methods

| Method | Class | Use Case |
|--------|-------|----------|
| **FineTune** | `FineTuneMethod` | Single-role supervised finetuning |
| **DFSFT** | `DiffusionForcingSFTMethod` | Diffusion-forcing SFT with inhomogeneous timesteps |
| **DMD2** | `DMD2Method` | Multi-role distribution matching distillation (student + teacher + critic) |
| **Self-Forcing** | `SelfForcingMethod` | Extends DMD2 for causal student rollouts |

### Launching Training (New Framework)

Training is launched via `torchrun` with a single YAML config:

```bash
torchrun --nproc_per_node <N_GPUS> \
  -m fastvideo.train.entrypoint.train \
  --config examples/train/<config>.yaml
```

### Example YAML Configs

| Config | Method | Description |
|--------|--------|-------------|
| `examples/train/finetune_wan2.1_t2v_1.3B_vsa_phase3.4_0.9sparsity.yaml` | FineTune | Wan 1.3B finetuning with VSA sparsity |
| `examples/train/distill_wan2.1_t2v_1.3B_dmd2.yaml` | DMD2 | Wan 1.3B distillation (student + teacher + critic) |
| `examples/train/dfsft_wan_causal_t2v_1.3B.yaml` | DFSFT | Causal Wan 1.3B diffusion-forcing SFT |
| `examples/train/self_forcing_wan_causal_t2v_1.3B.yaml` | Self-Forcing | Causal streaming distillation |

### Checkpointing (New Framework)

**CheckpointManager** (`utils/checkpoint.py`) saves via `torch.distributed.checkpoint`:

```
output_dir/
└─ checkpoint-{step}/
   ├─ dcp/                    # DCP state dict
   ├─ config.json             # resolved training config
   └─ .fastvideo_metadata.json
```

Checkpoint state includes: role model weights, per-role optimizers/schedulers,
CUDA RNG state, and callback state (e.g., EMA shadow weights).

### Config Structure

A YAML config defines the full training pipeline:

```yaml
models:
  student:
    _target_: fastvideo.train.models.wan.WanModel
    model_path: ...
    trainable: true
  teacher:  # optional, for distillation
    _target_: fastvideo.train.models.wan.WanModel
    model_path: ...
    trainable: false

method:
  _target_: fastvideo.train.methods.fine_tuning.FineTuneMethod
  # method-specific params...

training:
  distributed: { num_gpus: 8, tp_size: 1, sp_size: 8 }
  data: { data_path: ..., batch_size: 1 }
  optimizer: { lr: 1e-5, lr_scheduler: constant_with_warmup }
  loop: { max_train_steps: 1000 }
  checkpoint: { output_dir: ./outputs }
  tracker: { trackers: [wandb], project_name: ... }

callbacks:
  grad_clip:
    _target_: fastvideo.train.callbacks.GradNormClipCallback
    max_grad_norm: 1.0
  validation:
    _target_: fastvideo.train.callbacks.ValidationCallback
    validation_steps: 100
```

---

## Legacy Training Pipelines (`fastvideo/training/`)

> **Note:** Use the new `fastvideo/train/` framework for new work. This section
> is retained for reference on existing pipelines not yet migrated.

| Pipeline | Entrypoint | Use Case |
|----------|-----------|----------|
| Wan T2V finetune | `fastvideo/training/wan_training_pipeline.py` | Standard text-to-video finetune / LoRA |
| Wan I2V finetune | `fastvideo/training/wan_i2v_training_pipeline.py` | Image-to-video (first frame conditioned) |
| MatrixGame finetune | `fastvideo/training/matrixgame_training_pipeline.py` | Action-conditioned world model |
| LTX-2 finetune | `fastvideo/training/ltx2_training_pipeline.py` | LTX-2 architecture finetuning |
| Wan DMD distillation | `fastvideo/training/wan_distillation_pipeline.py` | Few-step distillation via DMD |
| Self-Forcing distill | `fastvideo/training/wan_self_forcing_distillation_pipeline.py` | Causal streaming distillation |

---

## Key Infrastructure

### W&B Integration
- **Tracker**: `fastvideo/training/trackers.py` — `WandbTracker` class
- **New framework tracker**: `fastvideo/train/utils/tracking.py` — `build_tracker()`
- **Env vars**: `WANDB_API_KEY`, `WANDB_BASE_URL`, `WANDB_MODE`

### Parallelism
- **SP** (Sequence Parallel): splits video frames across GPUs — `sp_size: N`
- **TP** (Tensor Parallel): splits model layers across GPUs — `tp_size: N`
- Typical configs: SP=2–8, TP=1–2

---

## Evaluation (for training runs)

Read `.agents/memory/evaluation-registry/README.md` for the full metric catalog.

**Quick summary for training agents:**
| Metric | When to Use | Trust |
|--------|-------------|-------|
| **Loss trajectory** | Every run, real-time from W&B | Medium |
| **SSIM** | When comparing against reference outputs | High |
| **FVD** | For benchmarking model quality (`benchmarks/fvd/`) | High |
| **LPIPS** | LoRA merge validation | Medium |
| **Human preference** | Major checkpoints | Highest |

---

## Common Workflows

| Task | Skill / SOP |
|------|-------------|
| Launch a training run | `.agents/skills/launch-experiment/SKILL.md` |
| Monitor a running experiment | `.agents/skills/monitor-experiment/SKILL.md` |
| Summarize final results | `.agents/skills/summarize-run/SKILL.md` |
| Full experiment lifecycle | `.agents/workflows/experiment-lifecycle.md` |
| Capture lessons from failures | `.agents/workflows/lesson-capture.md` |

---

## World Model–Specific Concepts

### Action Injection (MatrixGame)
The MatrixGame pipeline adds **action modules** to each DiT block, enabling
frame-level mouse/keyboard input conditioning. The action sequence is injected
per-frame alongside the latent video tokens.

### Causal Architecture
For streaming generation, the model uses **causal attention** (each frame only
attends to previous frames). This enables auto-regressive chunk-by-chunk
generation — critical for real-time interactive world models.

### Self-Forcing Distillation
A **data-free** distillation method where the student model is trained to
generate coherent video sequences by being forced to use its own previous
outputs (rather than ground-truth) as context. This produces models robust to
their own error accumulation during long auto-regressive generation.

### DMD Distillation (Distribution Matching Distillation)
Reduces inference steps from ~50 to 3–4 by training a student model to match
the output distribution of the teacher model. Uses a critic network to estimate
distribution divergence.

### Diffusion-Forcing SFT (DFSFT)
Supervised finetuning with **inhomogeneous timesteps** across chunks — each
chunk in a causal sequence can have a different noise level, training the model
to handle mixed-fidelity contexts.
