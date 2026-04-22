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
- Consistency finetuning with causal ODE initialization
- Action injection modules for interactive control

---

## Essential Reading (Training-Specific)

Read these **in order** before touching any training code:

| # | File | What You Learn |
|---|------|----------------|
| 1 | `docs/training/overview.md` | Training data flow: raw video → text embeddings + video latents → training |
| 2 | `docs/training/finetune.md` | All training arguments, parallelism (SP/TP), LoRA, validation settings |
| 3 | `docs/training/data_preprocess.md` | How to preprocess datasets into the expected format |
| 4 | `docs/design/overview.md` | Architecture: models, pipelines, configs, registry |

---

## Training Pipelines

Each pipeline is a Python entrypoint launched via `torchrun`:

| Pipeline | Entrypoint | Use Case |
|----------|-----------|----------|
| **Wan T2V finetune** | `fastvideo/training/wan_training_pipeline.py` | Standard text-to-video full finetune / LoRA |
| **Wan I2V finetune** | `fastvideo/training/wan_i2v_training_pipeline.py` | Image-to-video (condition on first frame) |
| **MatrixGame finetune** | `fastvideo/training/matrixgame_training_pipeline.py` | Action-conditioned world model training |
| **LTX-2 finetune** | `fastvideo/training/ltx2_training_pipeline.py` | LTX-2 architecture finetuning |
| **Wan DMD distillation** | `fastvideo/training/wan_distillation_pipeline.py` | Few-step distillation via DMD |
| **Self-Forcing distill** | `fastvideo/training/wan_self_forcing_distillation_pipeline.py` | Causal streaming distillation |
| **Base training** | `fastvideo/training/training_pipeline.py` | Base class — not called directly |

---

## Example Scripts

Ready-to-run training launches. Use these as starting templates:

### Finetuning
| Model | Script | Notes |
|-------|--------|-------|
| Wan T2V 1.3B | `examples/training/finetune/wan_t2v_1.3B/crush_smol/finetune_t2v.sh` | Smallest, fastest for testing |
| Wan T2V 1.3B LoRA | `examples/training/finetune/wan_t2v_1.3B/crush_smol/finetune_t2v_lora.sh` | Lightweight adapter |
| Wan I2V 14B | `examples/training/finetune/wan_i2v_14B_480p/crush_smol/finetune_i2v.sh` | Large I2V model |
| LTX-2 | `examples/training/finetune/ltx2/finetune_t2v.sh` | Alternative architecture |
| MatrixGame 2.0 | `examples/training/finetune/MatrixGame2.0/finetune_i2v.sh` | Action-conditioned |

### Distillation
| Method | Script | Notes |
|--------|--------|-------|
| DMD Wan | `scripts/distill/v1_distill_dmd_wan.sh` | Full distillation launch |
| DMD Wan + VSA | `scripts/distill/v1_distill_dmd_wan_VSA.sh` | With variable-step acceleration |
| Consistency (causal ODE init) | `examples/training/consistency_finetune/causal_ode_init/finetune_ode_init.sh` | Consistency tuning |

### Data Preprocessing
Each model directory includes a `preprocess_*` script. Always preprocess first:
```bash
# Example for Wan T2V
bash examples/training/finetune/wan_t2v_1.3B/crush_smol/preprocess_wan_data_t2v.sh
```

---

## Key Infrastructure

### W&B Integration
- **Tracker**: `fastvideo/training/trackers.py` — `WandbTracker` class
- **Env vars**: `WANDB_API_KEY`, `WANDB_BASE_URL`, `WANDB_MODE`
- **Summaries**: Saved to `<output_dir>/tracker/wandb/latest-run/files/wandb-summary.json`
- **Tests**: `fastvideo/tests/training/Vanilla/test_training_loss.py` (compares against reference summaries)

### Checkpointing
- **Save**: `fastvideo/training/training_utils.py:save_checkpoint()`
- **Load**: `fastvideo/training/training_utils.py:load_checkpoint()`
- **Distill save**: `training_utils.py:save_distillation_checkpoint()` (multi-model)
- **Format**: FSDP distributed checkpoint → converted to HF format

### Parallelism
- **SP** (Sequence Parallel): splits video frames across GPUs — `--sp_size N`
- **TP** (Tensor Parallel): splits model layers across GPUs — `--tp_size N`
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
the output distribution of the teacher model. Uses ODE pairs collected from
the teacher model as training targets.

---

## Quick-Start: Minimal Training Test

To verify the training infrastructure works, run the smallest possible experiment:

```bash
# 1. Download crush_smol dataset
bash examples/training/finetune/wan_t2v_1.3B/crush_smol/download_dataset.sh

# 2. Preprocess
bash examples/training/finetune/wan_t2v_1.3B/crush_smol/preprocess_wan_data_t2v.sh

# 3. Short training run (5 steps)
# Edit finetune_t2v.sh: set --max_train_steps 5
bash examples/training/finetune/wan_t2v_1.3B/crush_smol/finetune_t2v.sh
```
