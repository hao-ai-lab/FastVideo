---
name: add-reward-model
description: Use when adding reusable reward models under fastvideo/train/methods/rl/rewards for RLHF or online RL training.
---

# Add Reward Model

Use for reward models consumed by RL methods.

## Placement

- Put reusable reward code under `fastvideo/train/methods/rl/rewards/`.
- Expose public builders from `fastvideo/train/methods/rl/rewards/__init__.py`.
- Keep method-specific aggregation or advantage logic out of reward classes.

## Media Inputs

- Reward callables receive decoded media tensors.
- Accept single-frame tensors as `[B, C, H, W]` and multi-frame tensors as `[B, C, T, H, W]` when practical.
- Frame selection is reward-specific. Frame scorers such as PickScore and CLIPScore should explicitly select frame `0`; temporal rewards should inspect whichever frames they need.
- Return one scalar reward per prompt/sample.

## Attribution

- If code is ported or closely adapted from another repo, add a short comment or docstring naming the source file/function.
- Preserve SPDX headers used by FastVideo files.

## Tests

- Unit-test tensor layout handling without loading large reward checkpoints.
- Allow fake scorer injection for multi-reward tests.
- Test weighted reward aggregation and metric keys.
