# [RFC] Unified, YAML-Driven Training Architecture for Video Diffusion Models

## Summary

We propose a new **pluggable training architecture** in FastVideo that cleanly separates **models**, **training methods**, and **infrastructure** into independent, composable layers. A single YAML config file is all that is needed to train any supported model with any supported algorithm — no code changes required to mix and match.

This issue describes the design, current status, and open questions. **We welcome community feedback on the architecture, API surface, and planned extensions.**

---

## Motivation

Training video diffusion models today involves a tangle of concerns: model loading, noise scheduling, distillation algorithms, distributed strategies, checkpointing, and validation. Existing training scripts tend to hard-wire these together, making it painful to:

1. **Try a new distillation algorithm** on an existing model (requires forking the training loop).
2. **Add a new model** to an existing algorithm (requires re-implementing boilerplate).
3. **Switch distributed strategies** (FSDP, TP, SP) without touching algorithm code.
4. **Resume, checkpoint, and validate** uniformly across all combinations.

Our goal is a single, extensible training framework where each axis of variation is an independent plugin.

---

## Architecture Overview

```
YAML Config
    |
    v
+------------------+     +---------------------+     +------------------+
|   Models Layer   |     |   Methods Layer     |     | Infrastructure   |
|  (per-role)      |     |   (algorithm)       |     |   Layer          |
|                  |     |                     |     |                  |
| - ModelBase      |<----|  - TrainingMethod   |---->| - Trainer        |
| - CausalModelBase|     |  - single_train_step|     | - Callbacks      |
|                  |     |  - backward         |     | - Checkpoint     |
| Roles:           |     |  - optimizers       |     | - Tracker (W&B)  |
|  student         |     |                     |     | - Dataloader     |
|  teacher         |     | Algorithms:         |     |                  |
|  critic          |     |  DMD2, SelfForcing, |     | Distributed:     |
|                  |     |  SFT, DFSFT         |     |  HSDP, TP, SP    |
+------------------+     +---------------------+     +------------------+
```

### Three Layers

| Layer | Responsibility | Extension point |
|-------|---------------|-----------------|
| **Models** (`fastvideo/train/models/`) | Load transformer + scheduler, define `predict_noise`, `predict_x0`, `add_noise`, `backward`. Each training role (student/teacher/critic) is an independent instance. | Subclass `ModelBase` (or `CausalModelBase` for streaming). |
| **Methods** (`fastvideo/train/methods/`) | Implement the training algorithm: own role models, define `single_train_step` + `backward`, manage optimizers/schedulers. | Subclass `TrainingMethod`. |
| **Infrastructure** (`fastvideo/train/trainer.py`, `utils/`, `callbacks/`) | Training loop, gradient accumulation, distributed setup, checkpointing (DCP), W&B tracking, validation, EMA, grad clipping. | Add callbacks; everything else is shared. |

### YAML-Driven Configuration

Everything is configured declaratively. The `_target_` field selects the Python class to instantiate:

```yaml
models:
  student:
    _target_: fastvideo.train.models.wan.WanModel
    init_from: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    trainable: true
  teacher:
    _target_: fastvideo.train.models.wan.WanModel
    init_from: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    trainable: false
  critic:
    _target_: fastvideo.train.models.wan.WanModel
    init_from: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    trainable: true

method:
  _target_: fastvideo.train.methods.distribution_matching.dmd2.DMD2Method
  rollout_mode: simulate
  dmd_denoising_steps: [1000, 850, 700, 550, 350, 275, 200, 125]
  generator_update_interval: 5
  real_score_guidance_scale: 3.5
  # ...

training:
  distributed: { num_gpus: 8, sp_size: 1, tp_size: 1 }
  data: { data_path: ..., num_latent_t: 20, num_frames: 77 }
  optimizer: { learning_rate: 2.0e-6, betas: [0.0, 0.999] }
  loop: { max_train_steps: 4000 }
  checkpoint: { output_dir: outputs/my_run }

callbacks:
  grad_clip: { max_grad_norm: 1.0 }
  validation: { pipeline_target: ..., every_steps: 100 }
```

To switch from DMD2 to SFT, change the `method._target_` and remove the teacher/critic — no code changes.

---

## Model Abstraction

### `ModelBase` — Standard (Bidirectional) Models

Every role gets its own `ModelBase` instance owning a `transformer` and `noise_scheduler`. The base class defines:

- **`prepare_batch()`** — Convert raw dataloader output into forward-ready `TrainingBatch`.
- **`add_noise()`** — Apply forward-process noise at a given timestep.
- **`predict_noise()` / `predict_x0()`** — Run the transformer and return predictions.
- **`backward()`** — Backward pass that restores forward context (attention metadata, timesteps).
- **`init_preprocessors()`** — Lazy-load VAE, build dataloader (called only on the student).

### `CausalModelBase` — Streaming / Causal Models

Extends `ModelBase` with streaming inference primitives for causal video generation:

```python
class CausalModelBase(ModelBase):
    def clear_caches(self, *, cache_tag: str = "pos") -> None: ...
    def predict_noise_streaming(self, ..., cache_tag, store_kv, cur_start_frame) -> Tensor | None: ...
    def predict_x0_streaming(self, ..., cache_tag, store_kv, cur_start_frame) -> Tensor | None: ...
```

Key design: KV caches are **internal** to the model instance, keyed by `cache_tag`. The method controls when to store (`store_kv=True`) vs. read-only (`store_kv=False`), enabling block-by-block causal rollout during training.

---

## Supported Training Methods

### 1. DMD2 (Distribution Matching Distillation)

**Roles:** student (trainable) + teacher (frozen) + critic (trainable)

The student learns to generate clean video in few steps by matching the teacher's score function, with a critic network providing a learned fake-score baseline.

- **Rollout modes:**
  - `simulate` — Student starts from pure noise and iteratively denoises through the full step schedule.
  - `data_latent` — Student denoises from a single randomly-noised data sample.
- **Losses:** Generator loss (DMD gradient) + critic flow-matching loss, with alternating updates (`generator_update_interval`).

### 2. Self-Forcing (Causal DMD)

**Roles:** student (causal, trainable) + teacher (frozen) + critic (trainable)

Extends DMD2 for **streaming/causal video generation**. The key idea: during training, the student processes video in temporal chunks, using its own previously-denoised outputs as context for future chunks — simulating online autoregressive rollout.

- Video is split into blocks of `chunk_size` latent frames.
- Each block is denoised through the student's step schedule; a random early-exit step is sampled per block.
- After denoising a block, its output is fed back (with optional `context_noise`) as KV cache context for subsequent blocks via `predict_noise_streaming(store_kv=True)`.
- Supports SDE and ODE sampling during rollout.
- Selective gradient control: `enable_gradient_in_rollout`, `start_gradient_frame`.

### 3. Supervised Fine-Tuning (SFT)

**Roles:** student only

Standard flow-matching loss between predicted and ground-truth noise/x0.

### 4. Diffusion-Forcing SFT (DFSFT)

**Roles:** student only

SFT with **inhomogeneous (per-chunk) timesteps** — each temporal chunk in a video gets a different noise level. This trains the model to handle mixed-noise inputs, which is a prerequisite for causal/streaming inference where earlier frames are cleaner than later ones.

---

## Training Loop

The `Trainer` runs a standard loop with pluggable method and callbacks:

```
for step in range(start_step, max_steps):
    for accum_iter in range(grad_accum_steps):
        batch <- dataloader
        loss_map, outputs, metrics <- method.single_train_step(batch, step)
        method.backward(loss_map, outputs)

    callbacks.on_before_optimizer_step()   # grad clipping
    method.optimizers_schedulers_step()
    method.optimizers_zero_grad()
    callbacks.on_training_step_end()       # logging
    checkpoint_manager.maybe_save(step)
    callbacks.on_validation_begin()        # periodic inference
```

### Callbacks

- **GradNormClipCallback** — Per-module gradient norm logging + global clipping.
- **ValidationCallback** — Periodic inference sampling with configurable pipeline, sampling steps, and guidance scale.
- **EMACallback** — Exponential moving average of student weights.

### Checkpointing

- DCP (Distributed Checkpoint) format, compatible with FSDP/HSDP.
- Saves: model weights, optimizer states, scheduler states, RNG states (per role).
- Full resume support: auto-restores step counter and all RNG states.

---

## Current Status

| Component | Status |
|-----------|--------|
| Core framework (trainer, config, callbacks) | Implemented and tested |
| `WanModel` (Wan 2.1 T2V) | Implemented and tested |
| `WanGameModel` (WanGame 2.1 I2V) | Implemented and tested |
| `WanGameCausalModel` (streaming) | Implemented and tested |
| `WanCausalModel` (Wan T2V causal) | In progress |
| DMD2 method | Implemented and tested |
| Self-Forcing method | Implemented and tested |
| SFT method | Implemented and tested |
| DFSFT method | Implemented and tested |
| DCP checkpointing + resume | Implemented and tested |
| EMA callback | Implemented |
| Validation callback | Implemented and tested |
| Causal DMD inference pipeline | Implemented |

---

## Open Questions for Discussion

We'd love community input on the following:

### 1. Model Plugin API

The current `ModelBase` interface requires implementing 6 methods. Is this the right granularity?

- Should `prepare_batch` be split into separate concerns (noise sampling, timestep sampling, attention metadata)?
- Should `backward` be lifted out of the model and into the method/trainer?

### 2. Causal Streaming Interface

`CausalModelBase` adds `predict_noise_streaming` / `predict_x0_streaming` with cache management. Alternatives considered:

- **(a) Current:** Cache is internal to the model, keyed by `cache_tag`. Simple but couples cache lifecycle to model.
- **(b) External cache:** Method owns the cache dict, passes it into predict calls. More explicit but verbose.
- **(c) Context manager:** `with model.streaming_context(tag) as ctx: ...` — cleaner lifecycle but harder to compose.

Which approach do you prefer? Are there use cases that would break the current design?

### 3. Self-Forcing Training Strategy

The self-forcing rollout has several configurable knobs:

- **Early-exit sampling:** Random step per block vs. always last step (`last_step_only`). Random exit trains robustness but adds variance.
- **Context noise:** Re-noising completed blocks before feeding as context (`context_noise`). Improves robustness to prediction errors during inference.
- **Gradient scope:** Gradient only on exit-step prediction vs. through the entire rollout. Memory vs. signal quality trade-off.

What are your experiences with these trade-offs? Are there strategies we should consider?

### 4. Method Composition

Currently, methods are monolithic classes. Should we support composing methods (e.g., DFSFT pre-training followed by Self-Forcing distillation) within a single config? Or is sequential training with checkpoint handoff sufficient?

### 5. New Models and Methods

What models and training methods should we prioritize next?

- **Models:** HunyuanVideo, CogVideoX, other Wan variants?
- **Methods:** Consistency models, progressive distillation, reward-based fine-tuning?

### 6. Distributed Strategy

Currently supports HSDP (hybrid sharded data parallel) + TP + SP. Are there scenarios where the current distributed setup is insufficient? Should we add pipeline parallelism for very large models?

---

## How to Try It

```bash
# Install
uv pip install -e .[dev]

# Run DMD2 distillation on Wan 2.1
torchrun --nproc_per_node=8 -m fastvideo.train.entrypoint.train \
    --config examples/train/distill_wan2.1_t2v_1.3B_dmd2.yaml

# Run SFT fine-tuning
torchrun --nproc_per_node=8 -m fastvideo.train.entrypoint.train \
    --config examples/train/finetune_wan2.1_t2v_1.3B_vsa_phase3.4_0.9sparsity.yaml
```

Example configs are in
[`examples/train/`](https://github.com/hao-ai-lab/FastVideo/tree/main/examples/train).

---

## File Structure Reference

```
fastvideo/train/
  trainer.py                  # Training loop
  models/
    base.py                   # ModelBase, CausalModelBase ABCs
    wan/wan.py                # Wan 2.1 T2V model plugin
    wangame/wangame.py        # WanGame 2.1 I2V model plugin
    wangame/wangame_causal.py # WanGame causal (streaming) plugin
  methods/
    base.py                   # TrainingMethod ABC
    distribution_matching/
      dmd2.py                 # DMD2 distillation
      self_forcing.py         # Self-Forcing (causal DMD)
    fine_tuning/
      finetune.py             # Supervised fine-tuning
      dfsft.py                # Diffusion-forcing SFT
  callbacks/
    grad_clip.py              # Gradient clipping + norm logging
    validation.py             # Periodic inference validation
    ema.py                    # EMA weight averaging
  entrypoint/
    train.py                  # CLI entrypoint (torchrun)
  utils/
    config.py                 # YAML parser -> RunConfig
    builder.py                # build_from_config: model/method instantiation
    training_config.py        # TrainingConfig dataclass
    dataloader.py             # Dataset/dataloader construction
    optimizer.py              # Optimizer/scheduler construction
    checkpoint.py             # DCP save/resume
    tracking.py               # W&B tracker
```

---

## Related

- [RFC: Training Architecture](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/train/rfc.md) — Original internal design document.
- [Self-Forcing paper](https://arxiv.org/abs/2406.05477) — Chen et al., 2024.
- [DMD2 paper](https://arxiv.org/abs/2405.14867) — Yin et al., 2024.
- [Diffusion Forcing paper](https://arxiv.org/abs/2407.01392) — Chen et al., 2024.

---

**Feedback is highly welcome!** Please comment on this issue with your thoughts on the design, suggestions for improvement, or use cases that the current architecture doesn't cover well.
