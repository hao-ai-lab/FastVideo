# Training Infrastructure

FastVideo's training infrastructure (`fastvideo/train/`) is a YAML-driven
framework for training and distilling video diffusion models. A single config
file controls everything — models, algorithms, distributed strategy,
checkpointing, and validation — with no code changes needed to mix and match.

!!! note "Relationship to legacy training"
    This system replaces the older script-based training in `fastvideo/training/`.
    The legacy scripts still work for basic fine-tuning, but new development
    should use the config-driven system documented here.

---

## Quick Start

### Launch with the helper script

```bash
bash examples/train/run.sh examples/train/distill_wan2.1_t2v_1.3B_dmd2.yaml
```

The script auto-detects available GPUs and sets up `torchrun`. Override with
environment variables:

```bash
NUM_GPUS=4 NNODES=2 NODE_RANK=0 \
    MASTER_ADDR=10.0.0.1 MASTER_PORT=29501 \
    bash examples/train/run.sh my_config.yaml
```

### Launch directly with torchrun

```bash
torchrun --nproc_per_node=8 \
    fastvideo/train/entrypoint/train.py \
    --config examples/train/distill_wan2.1_t2v_1.3B_dmd2.yaml
```

### CLI flags

| Flag | Description |
|------|-------------|
| `--config` | Path to YAML config file (required) |
| `--resume-from-checkpoint` | Path to a DCP checkpoint directory to resume from |
| `--override-output-dir` | Override `training.checkpoint.output_dir` |
| `--dry-run` | Validate config and exit without training |

---

## Config Format

Every run is defined by a single YAML file with five top-level sections.
See `examples/train/example.yaml` for a fully-commented reference.

### `models` — Role-based model instances

Each entry defines a model role. The `_target_` field specifies the Python class
to instantiate:

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
    disable_custom_init_weights: true
```

Common model parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `_target_` | *(required)* | Python class path for the model |
| `init_from` | *(required)* | HuggingFace repo ID or local checkpoint path |
| `trainable` | `true` | Whether the model's parameters require gradients |
| `disable_custom_init_weights` | `false` | Skip custom weight initialization (use for teacher/critic) |
| `flow_shift` | `3.0` | Timestep shifting factor |
| `enable_gradient_checkpointing_type` | `null` | Gradient checkpointing (`"full"` or `null`) |

Which roles are needed depends on the training method:

| Method | Required roles |
|--------|---------------|
| Fine-tune (SFT) | `student` |
| Diffusion-Forcing SFT | `student` |
| DMD2 | `student`, `teacher`, `critic` |
| Self-Forcing | `student` (causal), `teacher`, `critic` |

### `method` — Training algorithm

Selects and configures the training algorithm:

```yaml
method:
  _target_: fastvideo.train.methods.distribution_matching.dmd2.DMD2Method
  rollout_mode: simulate
  dmd_denoising_steps: [1000, 750, 500, 250]
  generator_update_interval: 5
```

To switch algorithms, change `_target_` and adjust the method-specific keys.
See [Training Methods](#training-methods) for details on each algorithm.

### `training` — Typed infrastructure config

This section maps to typed dataclasses with defaults and validation:

```yaml
training:
  distributed:
    num_gpus: 8
    sp_size: 1            # sequence parallelism
    tp_size: 1            # tensor parallelism
    hsdp_replicate_dim: 1 # HSDP replication dimension
    hsdp_shard_dim: 8     # HSDP sharding dimension

  data:
    data_path: data/my_dataset
    train_batch_size: 1
    dataloader_num_workers: 4
    training_cfg_rate: 0.1  # classifier-free guidance dropout rate
    seed: 1000
    num_latent_t: 20
    num_height: 448
    num_width: 832
    num_frames: 77

  optimizer:
    learning_rate: 2.0e-6
    betas: [0.9, 0.999]
    weight_decay: 0.01
    lr_scheduler: constant  # constant, linear, cosine, polynomial
    lr_warmup_steps: 0

  loop:
    max_train_steps: 4000
    gradient_accumulation_steps: 1

  checkpoint:
    output_dir: outputs/my_run
    training_state_checkpointing_steps: 1000  # 0 = disabled
    checkpoints_total_limit: 3                # 0 = keep all

  tracker:
    project_name: my_project
    run_name: my_run

  model:
    weighting_scheme: uniform   # uniform, logit_normal, mode
    precondition_outputs: false
    enable_gradient_checkpointing_type: full

  vsa:
    sparsity: 0.0         # 0.0 = disabled
    decay_rate: 0.0
    decay_interval_steps: 0
```

### `callbacks` — Pluggable hooks

Callbacks run at specific points in the training loop (before/after optimizer
steps, at validation time, etc.):

```yaml
callbacks:
  grad_clip:
    max_grad_norm: 1.0

  ema:
    _target_: fastvideo.train.callbacks.ema.EMACallback
    decay: 0.9999
    start_iter: 0

  validation:
    _target_: fastvideo.train.callbacks.validation.ValidationCallback
    pipeline_target: fastvideo.pipelines.basic.wan.wan_pipeline.WanPipeline
    dataset_file: path/to/validation.json
    every_steps: 100
    sampling_steps: [4]
    guidance_scale: 5.0
```

See [Callbacks](#callbacks) for details on each callback.

### `pipeline` — Inference pipeline overrides

Optional overrides for the inference pipeline used during validation:

```yaml
pipeline:
  flow_shift: 8
```

---

## Training Methods

### Supervised Fine-Tuning (SFT)

Standard flow-matching loss. The simplest method — train the student to predict
noise (or clean x0) from noised data samples.

```yaml
models:
  student:
    _target_: fastvideo.train.models.wan.WanModel
    init_from: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    trainable: true

method:
  _target_: fastvideo.train.methods.fine_tuning.finetune.FineTuneMethod
  attn_kind: dense   # "dense" or "vsa"
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `attn_kind` | `"dense"` | Attention mode: `"dense"` (standard) or `"vsa"` (sparse) |

### Diffusion-Forcing SFT (DFSFT)

SFT with **per-chunk inhomogeneous timesteps** — each temporal chunk of the
video gets a different noise level. This is a prerequisite for training causal /
streaming models that must handle mixed-noise inputs.

```yaml
method:
  _target_: fastvideo.train.methods.fine_tuning.dfsft.DiffusionForcingSFTMethod
  chunk_size: 3
  min_timestep_ratio: 0.0
  max_timestep_ratio: 1.0
  attn_kind: dense
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | `3` | Latent frames per temporal chunk |
| `min_timestep_ratio` | `0.0` | Lower bound of timestep sampling range |
| `max_timestep_ratio` | `1.0` | Upper bound of timestep sampling range |
| `attn_kind` | `"dense"` | `"dense"` or `"vsa"` |

### DMD2 (Distribution Matching Distillation)

Distill a many-step teacher into a few-step student. The student learns to match
the teacher's score function, guided by a trainable critic network.

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
    disable_custom_init_weights: true
  critic:
    _target_: fastvideo.train.models.wan.WanModel
    init_from: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    trainable: true
    disable_custom_init_weights: true

method:
  _target_: fastvideo.train.methods.distribution_matching.dmd2.DMD2Method
  rollout_mode: simulate
  dmd_denoising_steps: [1000, 750, 500, 250]
  generator_update_interval: 5
  real_score_guidance_scale: 4.5

  fake_score_learning_rate: 8.0e-6
  fake_score_betas: [0.0, 0.999]
  fake_score_lr_scheduler: constant
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rollout_mode` | *(required)* | `"simulate"` (pure noise) or `"data_latent"` (from data) |
| `dmd_denoising_steps` | *(required)* | Timestep schedule for student rollout |
| `generator_update_interval` | `1` | Update student every N critic steps |
| `real_score_guidance_scale` | `1.0` | CFG scale for teacher predictions |
| `fake_score_learning_rate` | *(required)* | Critic optimizer learning rate |
| `fake_score_betas` | *(required)* | Critic optimizer Adam betas |
| `fake_score_lr_scheduler` | *(required)* | Critic LR scheduler type |

### Self-Forcing (Causal DMD)

Extends DMD2 for **streaming / causal video generation**. The student processes
video in temporal chunks, feeding its own denoised outputs as context for future
chunks — simulating autoregressive rollout during training.

Requires a causal model class (e.g., `WanCausalModel`) for the student:

```yaml
models:
  student:
    _target_: fastvideo.train.models.wan.wan_causal.WanCausalModel
    init_from: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    trainable: true

method:
  _target_: fastvideo.train.methods.distribution_matching.self_forcing.SelfForcingMethod
  rollout_mode: simulate
  dmd_denoising_steps: [1000, 750, 500, 250]
  student_sample_type: sde
  context_noise: 0.0
  enable_gradient_in_rollout: true
  start_gradient_frame: 0
```

Self-Forcing inherits all DMD2 parameters, plus:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `student_sample_type` | `"sde"` | `"sde"` or `"ode"` for intermediate steps |
| `same_step_across_blocks` | `false` | Use same exit timestep for all blocks |
| `last_step_only` | `false` | Always exit at the final denoising step |
| `context_noise` | `0.0` | Noise added to context frames (0 = clean) |
| `enable_gradient_in_rollout` | `true` | Enable backprop through rollout |
| `start_gradient_frame` | `0` | Frame index where gradients begin |

---

## Callbacks

Callbacks are pluggable hooks that run at specific points in the training loop.
Configure them under the `callbacks` section.

### GradNormClipCallback

Clips gradient norms before the optimizer step. Optionally logs per-module
gradient norms to the tracker.

```yaml
callbacks:
  grad_clip:
    max_grad_norm: 1.0      # 0.0 = disabled
    log_grad_norms: false
```

### EMACallback

Maintains an exponential moving average of the student's weights. The EMA
weights are automatically swapped in during validation.

```yaml
callbacks:
  ema:
    _target_: fastvideo.train.callbacks.ema.EMACallback
    decay: 0.9999
    start_iter: 0   # delay EMA updates until this iteration
```

The EMA callback owns its own state and checkpoints independently — EMA weights
are saved and restored automatically on resume.

### ValidationCallback

Runs inference with the trained model at regular intervals, saving generated
videos and logging them to the tracker (W&B).

```yaml
callbacks:
  validation:
    _target_: fastvideo.train.callbacks.validation.ValidationCallback
    pipeline_target: fastvideo.pipelines.basic.wan.wan_pipeline.WanPipeline
    dataset_file: path/to/validation.json
    every_steps: 100
    sampling_steps: [4]
    sampling_timesteps: [1000, 750, 500, 250]  # explicit timestep list
    guidance_scale: 5.0
    rollout_mode: parallel  # "parallel" or "streaming"
```

The validation dataset is a JSON file containing a list of prompt strings.
If EMA is enabled, validation automatically uses the EMA weights.

---

## Checkpointing and Resume

### Checkpoint format

Checkpoints use PyTorch Distributed Checkpoint (DCP) format, compatible with
FSDP/HSDP sharding. Each checkpoint saves:

- Model weights (all roles)
- Optimizer states (all roles)
- LR scheduler states
- RNG states (for exact reproducibility)
- EMA shadow weights (if enabled)
- Training step counter

Checkpoints are saved to `<output_dir>/checkpoint-<step>/`.

### Saving checkpoints

```yaml
training:
  checkpoint:
    output_dir: outputs/my_run
    training_state_checkpointing_steps: 1000  # save every N steps (0 = off)
    checkpoints_total_limit: 3                # rolling window (0 = keep all)
```

### Resuming training

Use `--resume-from-checkpoint` to resume from a specific checkpoint:

```bash
# Via the helper script
bash examples/train/run.sh my_config.yaml --resume outputs/my_run/checkpoint-2000

# Via torchrun directly
torchrun --nproc_per_node=8 \
    fastvideo/train/entrypoint/train.py \
    --config my_config.yaml \
    --resume-from-checkpoint outputs/my_run/checkpoint-2000
```

Or set it in the YAML:

```yaml
training:
  checkpoint:
    resume_from_checkpoint: outputs/my_run/checkpoint-2000
```

### Reproducibility

The training entrypoint enables deterministic mode automatically:

- `torch.backends.cudnn.benchmark = False`
- `torch.backends.cudnn.deterministic = True`
- `torch.use_deterministic_algorithms(True)`

A shared CUDA RNG generator is seeded from `training.data.seed` and threaded
through all random operations (noise sampling, timestep sampling, etc.).
Ranks within the same sequence-parallel group share a seed, ensuring identical
noise across SP shards.

---

## Distributed Training

The framework supports HSDP (Hybrid Sharded Data Parallel), Tensor Parallelism
(TP), and Sequence Parallelism (SP):

```yaml
training:
  distributed:
    num_gpus: 8
    sp_size: 1            # sequence parallelism group size
    tp_size: 1            # tensor parallelism group size
    hsdp_replicate_dim: 1 # number of HSDP replicas
    hsdp_shard_dim: 8     # number of HSDP shards
```

**HSDP** shards model parameters across `hsdp_shard_dim` GPUs and replicates
across `hsdp_replicate_dim` groups. The product
`hsdp_replicate_dim * hsdp_shard_dim` should equal `num_gpus`.

**Sequence parallelism** splits the sequence (video frames) across `sp_size`
GPUs within each data-parallel group. Useful for long videos that don't fit on a
single GPU.

---

## VSA (Variable Sparse Attention)

VSA progressively increases attention sparsity during training, reducing compute
while maintaining quality:

```yaml
training:
  vsa:
    sparsity: 0.9             # target sparsity level
    decay_rate: 0.03          # sparsity increment per decay interval
    decay_interval_steps: 1   # steps between sparsity increases
```

The effective sparsity at step `t` is
`min(sparsity, decay_rate * (t // decay_interval_steps))`.

---

## Extending the Framework

### Adding a new model

1. Create a new module under `fastvideo/train/models/` (e.g.,
   `fastvideo/train/models/mymodel/mymodel.py`).
2. Subclass `ModelBase` (or `CausalModelBase` for streaming models).
3. Implement the required methods:
   - `prepare_batch()` — convert raw dataloader output to `TrainingBatch`
   - `add_noise()` — forward-process noise addition
   - `predict_noise()` — run the transformer forward pass
   - `backward()` — backward pass with forward context restoration
4. Reference it in your YAML config:

```yaml
models:
  student:
    _target_: fastvideo.train.models.mymodel.mymodel.MyModel
    init_from: my-org/my-model
    trainable: true
```

### Adding a new training method

1. Create a new module under `fastvideo/train/methods/`.
2. Subclass `TrainingMethod`.
3. Implement the required methods:
   - `single_train_step()` — one forward pass returning losses, outputs, metrics
   - `get_optimizers()` — return optimizer list
   - `get_lr_schedulers()` — return scheduler list
4. Reference it in your config:

```yaml
method:
  _target_: fastvideo.train.methods.my_method.MyMethod
  my_param: 42
```

Method-specific parameters are accessible via `self.method_config` (a plain
dict).

### Adding a new callback

1. Create a new module under `fastvideo/train/callbacks/`.
2. Subclass `Callback`.
3. Override the hooks you need: `on_train_start`, `on_training_step_end`,
   `on_before_optimizer_step`, etc.
4. Optionally implement `state_dict()` / `load_state_dict()` for checkpoint
   persistence.
5. Add it to your config:

```yaml
callbacks:
  my_callback:
    _target_: fastvideo.train.callbacks.my_callback.MyCallback
    my_param: 42
```

---

## File Structure

```
fastvideo/train/
  entrypoint/
    train.py                  # CLI entrypoint (torchrun)
  trainer.py                  # Training loop orchestrator
  models/
    base.py                   # ModelBase, CausalModelBase ABCs
    wan/
      wan.py                  # Wan 2.1 T2V model
      wan_causal.py           # Wan causal (streaming) model
  methods/
    base.py                   # TrainingMethod ABC
    distribution_matching/
      dmd2.py                 # DMD2 distillation
      self_forcing.py         # Self-Forcing (causal DMD)
    fine_tuning/
      finetune.py             # Supervised fine-tuning
      dfsft.py                # Diffusion-forcing SFT
  callbacks/
    callback.py               # Callback ABC and CallbackDict
    grad_clip.py              # Gradient clipping + norm logging
    ema.py                    # EMA weight averaging
    validation.py             # Periodic inference validation
  utils/
    config.py                 # YAML parser -> RunConfig
    training_config.py        # Typed config dataclasses
    builder.py                # Model/method instantiation
    optimizer.py              # Optimizer/scheduler construction
    checkpoint.py             # DCP save/resume
    dataloader.py             # Dataset/dataloader construction
    tracking.py               # W&B tracker
```

---

## Related Docs

- [Training Architecture](../design/training_architecture.md) — design
  rationale, model/method abstractions, and open questions.
- [Training Overview](overview.md) — data requirements and preprocessing.
- [Data Preprocessing](data_preprocess.md) — how to prepare datasets.
- [Config Reference](../../examples/train/configs/example.yaml) — fully-commented
  YAML config with all fields and defaults.
