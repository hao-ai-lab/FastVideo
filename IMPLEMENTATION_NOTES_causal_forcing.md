# Causal-Forcing in FastVideo: Clean-History Teacher Forcing + Causal Consistency Distillation

Adds two of the critical Wan-parity gaps identified vs `thu-ml/Causal-Forcing`:
**Stage-1 clean-history Teacher Forcing (TF)** and **Stage-2b Causal Consistency
Distillation (CD / Causal-Forcing++)**. Implemented in the modular `fastvideo/train/`
stack, reusing existing abstractions; the legacy `fastvideo/training/` stack is
untouched.

## What was added

### 1. DiT: clean-history teacher-forcing path
`fastvideo/models/dits/causal_wanvideo.py`
- `_prepare_teacher_forcing_mask(...)` — ported from Causal-Forcing
  `wan/modules/causal_model.py:569`. Builds a FlexAttention mask over the
  concatenated `[clean | noisy]` sequence (length `2 * num_frames * frame_seqlen`):
  a noisy block attends to its own noisy tokens plus the **clean** context of all
  strictly previous blocks (+ diagonal).
- `_forward_train(..., clean_x=None, aug_t=None)` — when `clean_x` is provided:
  patch-embeds the clean latents, prepends them to the noisy tokens, tiles RoPE and
  the per-frame timestep modulation so clean frame *i* and noisy frame *i* share the
  same position / embedding (clean frames are time-embedded at `aug_t`, default 0),
  runs the blocks under the TF mask, then drops the clean half before `norm_out`
  (loss is denoise-only on the current/noisy tokens). The diffusion-forcing path is
  unchanged. A separate `teacher_forcing_block_mask` is cached so the two masks never
  collide.

### 2. Wrapper plumbing
`fastvideo/train/models/wan/wan.py`
- `predict_noise(..., clean_x=None, aug_t=None)` and `_build_distill_input_kwargs(...)`
  forward `clean_x`/`aug_t` to the transformer (permuting `clean_x` to `[B,C,F,H,W]`).
  The streaming/KV-cache path is untouched.

### 3. Teacher-Forcing SFT method (Stage 1)
`fastvideo/train/methods/fine_tuning/tfsft.py` — `TeacherForcingSFTMethod`, a thin
subclass of `DiffusionForcingSFTMethod`. Identical objective (inhomogeneous per-chunk
timesteps, flow MSE, bsmntw weighting) except it passes `clean_x=clean_latents`. The
only change to `dfsft.py` is extracting the predict call into an overridable
`_predict_noise(...)` hook (behavior-preserving for DFSFT).

### 4. Causal Consistency Distillation method (Stage 2b)
`fastvideo/train/methods/consistency_model/causal_cd.py` —
`CausalConsistencyDistillationMethod`, ported from Causal-Forcing
`model/naive_consistency.py`. Per step:
1. Sample a discrete index over a `discrete_cd_N`-step flow-match schedule → `(t, t_next)`.
2. **Teacher** (frozen, teacher-forced) does a single CFG Euler step
   `latent_t → latent_t_next = latent_t - dt·[v_uncond + g·(v_cond - v_uncond)]`,
   `dt = (t - t_next)/num_train_timesteps`.
3. **Student** (trainable, teacher-forced) predicts `x0` at `t`.
4. **EMA** student (frozen, teacher-forced) predicts `x0` at `t_next`.
5. Loss `= MSE(x0_t, x0_t_next)`; EMA is lerp-updated after each optimizer step.

Roles: `student` (trainable), `teacher` and `ema` (frozen) — all initialized from the
same checkpoint, matching the reference's three-network setup.

### 5. Frame-wise DFSFT
A frame-wise variant of the existing diffusion-forcing SFT: each frame gets its own
independent noise level (block size 1) instead of sharing one level across a chunk.
No new method — `DiffusionForcingSFTMethod` with `chunk_size: 1` against a
`WanCausalModel` constructed with `num_frames_per_block: 1`. The model gained a
`num_frames_per_block` override kwarg (`fastvideo/train/models/wan/wan_causal.py`)
that the YAML `models.<role>` block maps to. Example config
`examples/train/configs/fine_tuning/wan/dfsft_causal_t2v_framewise.yaml`; smoke test
`fastvideo/tests/train/methods/test_wan_causal_dfsft_framewise.py`.

### 6. Configs, tests, registration
- Smoke tests (mirror `test_wan_causal_dfsft.py`, real Wan-1.3B + tiny synthetic data):
  `fastvideo/tests/train/methods/test_wan_causal_tfsft.py`,
  `.../test_wan_causal_cd.py`, `.../test_wan_causal_dfsft_framewise.py`, with fixtures
  `fastvideo/tests/train/fixtures/wan_causal_t2v_{tfsft,causal_cd,dfsft_framewise}_min.yaml`.
- Example run configs: `examples/train/configs/fine_tuning/wan/tfsft_causal_t2v.yaml`,
  `.../fine_tuning/wan/dfsft_causal_t2v_framewise.yaml`,
  `examples/train/configs/consistency_model/wan/causal_cd_t2v.yaml`.
- Registered in `fastvideo/train/methods/__init__.py`,
  `.../fine_tuning/__init__.py`, `.../consistency_model/__init__.py`.

## Running the smoke tests
```bash
conda activate FastVideo_kaiqin
export PYTHONPATH=$PWD   # from the FastVideo_cf clone
pytest fastvideo/tests/train/methods/test_wan_causal_tfsft.py -q
pytest fastvideo/tests/train/methods/test_wan_causal_cd.py -q
pytest fastvideo/tests/train/methods/test_wan_causal_dfsft_framewise.py -q
```
All require CUDA and the cached `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` weights. Each runs
one real train step and asserts finite loss + nonzero student gradients reaching the
first transformer block (CD additionally asserts a frozen teacher and a working EMA
update; the frame-wise test asserts the block-size override reached the transformer).
**Status: all pass on GB200.**

For W&B logging on a real run, set `WANDB_API_KEY` in the environment (do not commit
keys) and add a `training.tracker` block (see the example configs).

## Assumptions & deviations from the reference
- **Discrete CD schedule.** The reference builds its own `FlowMatchScheduler(shift=5,
  extra_one_step=True).set_timesteps(discrete_cd_N)`. Here we sub-sample `discrete_cd_N`
  points from the student's `FlowMatchEulerDiscreteScheduler.timesteps` and reuse the
  reference's Euler convention `dt = (t - t_next)/num_train_timesteps`. The
  shift-vs-linear sigma parameterization is therefore approximate in exactly the same
  way the reference's `/1000` Euler step is — internally consistent, but not a
  bit-exact reproduction of the reference scheduler. Tune `discrete_cd_N`/`flow_shift`
  for a real run.
- **EMA as an explicit role, not a deepcopy.** The reference keeps `generator_ema` and
  updates it via an external EMA tracker (`ema_model.copy_to`). FSDP `fully_shard`
  forbids `deepcopy`, so the EMA is a third role model loaded from the same checkpoint
  and lerp-updated in place (`optimizers_schedulers_step`). This is FSDP-safe and the
  EMA shards align with the student's. It does cost a third full model copy (as the
  reference also effectively does). It is **not** integrated with FastVideo's
  `EMACallback` (which is export/inference-only); a production setup may want to unify
  these.
- **`aug_t` noise augmentation** of the clean context (reference
  `noise_augmentation_max_timestep`) is plumbed through the DiT (`aug_t`) but the TF
  method passes `aug_t=None` (clean context at timestep 0), matching the reference's
  default. CD also uses clean (un-augmented) context.
- **Chunk vs frame granularity** is the `num_frames_per_block` knob (1 = frame,
  3 = chunk); TF, CD and DFSFT all honor it. `WanCausalModel` accepts a
  `num_frames_per_block` override (set from the `models.<role>` YAML block) so a run
  can pick the block size without a separate checkpoint; the frame-wise DFSFT recipe
  uses it. There is still no shipped Wan framewise *checkpoint*, only the recipe.
- **CFG uncond text** for the teacher comes from the student's negative-prompt
  conditioning (`unconditional_dict`), as in the reference.

## Not implemented (out of scope for this change)
- Stage-2a Causal **ODE** init / `blockwise_kv` data curation (the CD branch is the
  ODE-data-free alternative).
- Stage 3 asymmetric **DMD** changes, 1/2-step few-step (ASD first-chunk), SiD/GAN.
- Native frame-wise **I2V** architecture, Rolling-Forcing long video, TAEHV/TRT decode.
- `--use_ema` selection at standalone inference.
See `../CAUSAL_FORCING_PARITY_GAP_ANALYSIS.md` for the full gap list and priorities.
