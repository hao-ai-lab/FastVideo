---
name: add-rl-method
description: Use when adding or modifying an RL/RLHF method under fastvideo/train/methods/rl, including DiffusionNFT-like methods.
---

# Add RL Method

Use for new RL methods in the modular `fastvideo/train` stack.

## Required Shape

- Add the method under `fastvideo/train/methods/rl/`.
- Subclass `TrainingMethod`.
- Keep model-family logic in `ModelBase` wrappers.
- Decode generated latents through `ModelBase.decode_latents`; add that hook to the new model wrapper instead of decoding inside the RL method.
- Use `fastvideo/train/methods/rl/common/sampling.py` for generation unless the method has a documented reason to avoid sampling.
- Use `fastvideo/train/methods/rl/common/prompt_sampling.py` for reusable grouped prompt sampling patterns such as DiffusionNFT K-repeat.
- Use `fastvideo/train/methods/rl/rewards/` for reward models.

## Optimization

- Return `manages_optimization() == True` only when the method must own a nonstandard outer/inner loop.
- If using managed optimization, implement `managed_train_step(data_stream, iteration)`.
- Existing trainer callbacks, checkpointing, tracking, and validation should still work.

## Config

- Put method knobs under `method`.
- Put sampler knobs under `method.sampling`.
- Do not put scheduler or trajectory policy into model configs.
- Do not split a diffusers-style scheduler from its built-in `step()` solver in YAML; use `trajectory` only for higher-level ODE vs re-noise behavior.
- Avoid fixed timestep lists in examples unless reproducing a known baseline; prefer scheduler-generated defaults.

## Tests

- Add fake-model tests for sampler/method behavior.
- Add config parse tests for the public YAML.
- Confirm existing train methods stay on the default Trainer path.
