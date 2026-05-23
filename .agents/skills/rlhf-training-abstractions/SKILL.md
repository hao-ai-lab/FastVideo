---
name: rlhf-training-abstractions
description: Use when changing FastVideo RLHF/RL training infrastructure, especially sampler, reward, scheduler trajectory, or method boundaries under fastvideo/train.
---

# RLHF Training Abstractions

Use this skill before editing RLHF-style training code in `fastvideo/train`.

## Boundaries

- RL methods live under `fastvideo/train/methods/rl/` and own algorithm logic: reward collection, advantage computation, policy loss, KL/reference terms, and optimizer cadence.
- Rewards live under `fastvideo/train/methods/rl/rewards/` and must be reusable across RL methods.
- RL methods pass decoded media to rewards; each reward decides whether to use the first frame, sampled frames, or the full video.
- Sampling lives under `fastvideo/train/methods/rl/common/` and must use `ModelBase` primitives plus scheduler math, not model-family inference pipelines.
- Model wrappers under `fastvideo/train/models/` own model-specific forward details.
- Model wrappers also own model-specific latent decoding via `ModelBase.decode_latents`; RL methods should not reach into VAE normalization internals.
- Shared RL helpers such as K-repeat prompt sampling belong under `fastvideo/train/methods/rl/common/` when they are reusable across RL methods.

## Anti-Patterns

- Do not bind RL methods to inference pipeline classes such as `WanDMDPipeline`.
- Do not hardcode timestep lists in a method when the scheduler can generate them.
- Do not put reward-model code inside one RL method.
- Do not make existing non-RL methods use method-managed optimization unless explicitly requested.

## Sampling Policy

- Prefer YAML-configured `method.sampling` with `scheduler`, `trajectory`, `num_steps`, `timesteps`, and `sigmas`.
- Treat diffusers-style scheduler classes as owning both the timestep schedule and their `step()` update rule; avoid a separate `solver` field unless a new sampler truly implements solver math outside the scheduler object.
- Missing `timesteps` means “ask the scheduler”; explicit `timesteps` or `sigmas` are overrides.
- ODE-style trajectories should not re-noise between denoising steps.
- SDE/re-noise behavior must be explicit in config.

## Validation

- Run focused local tests for sampler config and Trainer opt-in behavior.
- Verify existing train methods still report `manages_optimization() == False`.
- Keep fixed-prompt validation helpers in `fastvideo/train/methods/rl/common/validation.py` so new RL methods can reuse sharding and captions.
- Test distributed prompt grouping helpers separately from heavyweight model loading.
- Run `pre-commit run --files <changed paths>`; respect configured excludes.
