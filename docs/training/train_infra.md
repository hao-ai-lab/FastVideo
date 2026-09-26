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
| `attention_backend` | `null` | Optional role-local backend for Wan models (for example `ATTN_QAT_TRAIN`); overrides the process default only while this role's transformer is built |

Which roles are needed depends on the training method:

| Method | Required roles |
|--------|---------------|
| Fine-tune (SFT) | `student` |
| Diffusion-Forcing SFT | `student` |
| DMD2 | `student`, `teacher`, `critic` |
| TDM | `student`, `teacher`, `critic` |
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
    trackers: []  # options: none, wandb, swanlab, jsonl
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

`training.data.data_path` can also mix multiple preprocessed datasets by using a mapping from dataset path to repeat count:

```yaml
training:
  data:
    data_path:
      data/zeldam2-clean: 1
      data/multi3d_games: 2
```

The repeat count duplicates that dataset's parquet file list before shuffling/sampling, so the example above trains with roughly twice as much `multi3d_games` exposure as `zeldam2-clean`. Paths are just suggested locations; use any local path that contains a FastVideo preprocessed parquet dataset.

See [Training Trackers](trackers.md) to configure Weights & Biases or SwanLab,
including SwanLab installation and authentication.

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

Registered transformer linear-quantization configs can also be selected by
name. For example, the LTX-2 NVFP4-QAT recipe applies real FP4 forward GEMMs
with a straight-through-estimator backward to its deployment-targeted
attention/FFN projections:

```yaml
pipeline:
  dit_config:
    quant_config: nvfp4_qat_train
```

The LTX-2 recipe in
`examples/train/configs/overfit_ltx2_t2v_nvfp4_qat.yaml` combines that linear
configuration with `models.student.attention_backend: ATTN_QAT_TRAIN` for
video-attention forward/backward. On sm120, its validation callback temporarily
switches those layers to `ATTN_QAT_INFER`.
On GB200, set `callbacks.validation.attn_qat_infer: false` to keep validation on
the train-time QAT backend; the inference kernel is sm120-only.

User-adaptable LTX-2 fine-tuning recipes (full, LoRA, and NVFP4 QAT) live in
`examples/train/configs/fine_tuning/ltx2/`, alongside the other model
families under `examples/train/configs/fine_tuning/`.

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
| `min_timestep_ratio` | `0.0` | Lower bound for randomly sampled teacher/critic score timesteps |
| `max_timestep_ratio` | `1.0` | Upper bound for randomly sampled teacher/critic score timesteps |
| `fake_score_learning_rate` | *(required)* | Critic optimizer learning rate |
| `fake_score_betas` | *(required)* | Critic optimizer Adam betas |
| `fake_score_lr_scheduler` | *(required)* | Critic LR scheduler type |

### TDM (Trajectory Distribution Matching)

Ports Trajectory Distribution Matching to FastVideo's modular trainer. The
original TDM paper and demo target CogVideoX-2B with diffusion notation; the
FastVideo implementation adapts the objective to Wan flow matching. Production
code uses Wan's forward process:

```text
x_sigma = (1 - sigma) * x0 + sigma * eps
x0_hat = x_sigma - sigma * model_output
```

It keeps the TDM role structure: a few-step trainable student generates a
trajectory, a trainable fake-score critic learns from generated trajectory
points, and a frozen teacher supplies real-score guidance for the generator
target.

```yaml
models:
  student:
    _target_: fastvideo.train.models.wan.WanModel
    init_from: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    trainable: true
    lora:
      enable: true
      rank: 16
      alpha: 32
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
    lora:
      enable: true
      rank: 16
      alpha: 32

method:
  _target_: fastvideo.train.methods.distribution_matching.tdm.TDMMethod
  rollout_mode: simulate
  tdm_denoising_steps: [1000, 750, 500, 250]
  generator_update_interval: 5
  real_score_guidance_scale: 4.5
  student_sample_type: sde
  noise_interval_mode: separate
  use_randmid: false
  max_grad_norm: 1.0
  cfg_uncond:
    text: negative_prompt
    on_missing: error

  fake_score_learning_rate: 8.0e-6
  fake_score_betas: [0.0, 0.999]
  fake_score_lr_scheduler: constant
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rollout_mode` | *(required)* | Currently must be `"simulate"` |
| `tdm_denoising_steps` | *(required)* | Few-step student trajectory schedule; mapped sigmas must start at scheduler terminal noise and strictly decrease |
| `warmup_steps` | `200` for Wan, `0` otherwise | Regression-warmup updates that fit the student to the guidance-combined teacher x0 at its own rollout states before TDM starts; the critic takes no updates and stays out of the optimizers, LR schedulers, and grad-clip targets while warmup is active |
| `tdm_step_ladder` | *(unset)* | Staged step counts, `[{denoising_steps: [...], until_iteration: N}]`; counts must strictly decrease, boundaries strictly increase, and only the last stage may omit `until_iteration`. Validation and inference adopt the final stage's schedule |
| `student_sample_type` | `"sde"` | `"sde"` re-noises each predicted x0; `"ode"` carries effective flow noise |
| `noise_interval_mode` | `"separate"` | Fake-score noising target selection mode; see note below |
| `use_randmid` | `false` | Randomly sample the intermediate sigma between the source point and the next trajectory sigma when enabled |
| `snr_clip` | `5.0` | Clip the flow-SNR fake-score weight |
| `importance_weight_clip` | `10.0` | Clip mixed-noise importance weights |
| `normalize_generator_delta` | `true` | Divide each sample's generator loss by its teacher-guidance magnitude |
| `tdm_vsa_apply_to` | `"student"` | Which roles run sparse attention when a sparse backend is configured: `"student"` (the validated H3 recipe) or `"all"` (critic and teacher sparse as well; their `models.<role>.attention_backend` must then be the matching sparse backend) |
| `use_huber` | `false` | Use the reference pseudo-Huber expression for the generator loss; fake-score training remains MSE |
| `huber_c` | `0.001` | Huber delta when `use_huber=true` |
| `use_pseudo_huber` | `false` | Use the paper Eq. 11 pseudo-Huber surrogate (`sqrt(||pred - target||_2^2 + c^2) - c`, `c = 0.00054*sqrt(d)` with `d` the flattened per-sample latent size); skips DMD delta normalization; mutually exclusive with `use_huber` |
| `max_grad_norm` | `1.0` | Clip student and critic gradients inside TDM's ordered optimizer phases; set to zero to disable |

See `examples/train/configs/distribution_matching/wan/tdm_t2v_lora.yaml` for a
complete Wan LoRA config. Treat this as a Wan adaptation of TDM, not exact
CogVideoX reference parity. TDM follows the reference implementation's rollout
gradient behavior: generated rollout history is not backpropagated through, and
only the student prediction used by the generator loss carries gradients. Each
training step first backpropagates and applies the fake-score critic update,
then resamples the trajectory point and proposal noise from the same detached
student trajectory and recomputes the generator loss against the updated critic
before applying the student update.

Fake-score training samples a source point from the generated trajectory. In
`separate` and `to_terminal` modes, source points are sampled randomly and
independently per batch element. With `use_randmid: true`, TDM then samples an
intermediate sigma in
`[sigma_next, sigma_source)`; otherwise the intermediate sigma is
`sigma_next`. For `noise_interval_mode: separate`, the target is sampled in
`[sigma_intermediate, sigma_source)`. For `noise_interval_mode: to_terminal`,
the target is sampled in `[sigma_intermediate, sigma_terminal)`, so it may be
any scheduler point in that interval. The exact terminal `sigma=1.0` endpoint
is excluded because flow-SNR weighting gives it zero fake-score weight.

For classifier-free teacher guidance, the shipped Wan recipes encode the
model's negative prompt and require it to be present. Zero text embeddings are
not an equivalent unconditional condition.

### TDM on MiniMax H3 (joint video + audio)

MiniMax H3 denoises one packed video-and-audio sequence, and its TDM port
trains both modalities in a single joint pass. The student and critic carry
rank-16 LoRA adapters over the frozen base and the teacher runs the frozen base
without adapters. H3 is guidance-distilled, so `real_score_guidance_scale`
stays `1.0` and there is deliberately no `method.cfg_uncond` block: the model
has no unconditional branch, and TDM uses the conditional teacher x0 directly.

The plugin implements the per-modality contract the method loops over
(`tdm_modalities`, `tdm_clean_latents`, `tdm_sigma_grid`, `tdm_predict_x0`,
and friends). H3 conventions that differ from Wan:

- **Sigma grid**: `shift * u / (1 + (shift - 1) * u)` with `u = label / 1000`,
  video shift `12` and audio shift `3`; the same integer labels drive both
  modalities, each mapped through its own shift.
- **Model time** is `1 - sigma` (no scheduler sigma-table lookup).
- **Velocity is data-ward**: the plugin negates the transformer output so the
  base conversion `x0 = x_t - sigma * pred_noise` is exactly
  `x0 = x_t + sigma * v`.
- **Losses are summed per modality**; one joint transformer forward serves the
  rollout, the fake-score update, and the generator update.

```yaml
models:
  student:
    _target_: fastvideo.train.models.minimax_h3.MiniMaxH3Model
    init_from: <MiniMax-H3 model dir>
    trainable: true
    attention_backend: TORCH_SDPA
    enable_gradient_checkpointing_type: full
    lora:
      enable: true
      rank: 16
      alpha: 16
      target_modules: [to_q, to_k, to_v, to_out]
  # teacher: trainable false, no lora block
  # critic:  trainable true, same lora block as the student

method:
  _target_: fastvideo.train.methods.distribution_matching.tdm.TDMMethod
  real_score_guidance_scale: 1.0
  warmup_steps: 50
  generator_update_interval: 1
  tdm_denoising_steps: [1000, 750, 500, 250]
  fake_score_learning_rate: 1.0e-4
```

The shipped example is
`examples/train/configs/distribution_matching/overfit_minimax_h3_t2va_tdm.yaml`.
The standalone H3 recipe used critic `1e-4` and generator `2e-5` (the critic
rate times a `0.2` generator scale), which the example encodes as the training
optimizer LR `2e-5` plus `fake_score_learning_rate: 1e-4`.

#### Geometry and memory

TDM prepares `latents_source="zeros"`, so the dataset's latent values and
shapes are unused: the trajectory geometry comes from
`training.data.num_height` / `num_width` / `num_latent_t` / `num_frames`, and
changing the canvas is a config override rather than a new preprocessed asset.
On four GB200s the joint stack fits at `480x832x124` (about 22-30 s per step);
the dense path also fits at `768x1344x124` (about 95-130 s per step), while VSA
at `768x1344` exceeds 184 GiB per GPU. Sequence parallelism is not a workaround
for this loader (`sp_size: 2` fails in FSDP device-mesh setup).

#### Video-sparse attention

Set `models.student.attention_backend: VIDEO_SPARSE_ATTN_H3` and
`vsa.sparsity` to run the student through H3's tile-64 Triton block-sparse
kernels. The TDM call sites already produce the standalone-validated
student-only split (the student uses `attn_kind="vsa"` while the critic and
teacher stay dense), so that recipe needs no extra knob. The kernel requires
bf16 Q/K/V, and the H3 VSA backend casts at the kernel boundary because the
training forward can carry fp32 through the QK-norm path.

Measured on four GB200s at `480x832x124`, 200-step overfit, sparsity `0.5`:
VSA reads `std 56.1 / sharpness 37.8` against a same-geometry dense control at
`std 55.7 / sharpness 43.2`, with visually equivalent samples, so VSA holds the
dense quality band at about 12 percent lower sharpness.
`method.tdm_vsa_apply_to: all` extends sparse attention to the critic and
teacher as well (set their `models.<role>.attention_backend` to the sparse
backend too) for the standalone's aggressive sparsity-`0.9` corner.

#### In-process validation

Running the H3 validation pipeline between training steps leaves the
autograd/CUDA stream state inconsistent, and the next training backward raises
`opt_ready_stream && opt_parent_stream`. Validate once at the final step
(`callbacks.validation.run_at_start: false`,
`callbacks.validation.every_steps: <max_train_steps>`) or sample from
checkpoints in a separate process.

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

### Streaming Long Tuning

`StreamingLongTuningMethod` extends Self-Forcing for LongLive-style rollouts. It
keeps a streaming state, generates overlapping chunks, and trains only the new
frames while preserving context from earlier chunks.

For the MatrixGame2/Zelda world-model example, self-forcing and long tuning are
separate runs: first train or load the 1k-step self-forcing checkpoint using
`examples/train/scenario/worldmodel/zelda/self_forcing_causal_i2v.yaml`,
then run
`examples/train/scenario/worldmodel/zelda/streaming_long_tuning_causal_i2v.yaml`
from that checkpoint for the 3k-step streaming long-tuning stage.

```yaml
method:
  _target_: fastvideo.train.methods.distribution_matching.streaming_long_tuning.StreamingLongTuningMethod
  streaming_chunk_size: 9
  streaming_max_length: 39
  streaming_fixed_overlap_latents: 3
  streaming_reencode_overlap_anchor: true
  streaming_anchor_inject_k: 1
  streaming_require_full_blocks: true
  multi_phased_distill_schedule:
    - stage: streaming_long
      start_step: 0
      end_step: 3000
      num_latent_t: 39
      streaming_training: true
```

See
`examples/train/scenario/worldmodel/zelda/streaming_long_tuning_causal_i2v.yaml`
for a complete MatrixGame2/Zelda configuration.

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
videos and logging them to the configured tracker. With `jsonl`, metrics are
written to `output_dir/tracker/metrics.jsonl` and artifact metadata to
`output_dir/tracker/artifacts.jsonl`.

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
      tdm.py                  # Trajectory Distribution Matching
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
    tracking.py               # W&B / SwanLab / JSONL trackers
```

---

## Related Docs

- [Training Architecture](../design/training_architecture.md) — design
  rationale, model/method abstractions, and open questions.
- [Training Overview](overview.md) — data requirements and preprocessing.
- [Data Preprocessing](data_preprocess.md) — how to prepare datasets.
- [Config Reference](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/train/configs/example.yaml) — fully-commented
  YAML config with all fields and defaults.
