---
date: 2026-09-24
experiment: Wan-VACE text-encoder parity against Diffusers
category: porting
severity: important
---

# Wan-VACE: `update_model_arch` resets T5 tokenizer kwargs

## What Happened

VACE prompt embeddings diverge from Diffusers even when `padding="max_length"` is set on the pipeline config's text encoder arch.

## Root Cause

`TextEncoderLoader.update_model_arch()` re-runs `T5ArchConfig.__post_init__`, which rebuilds `tokenizer_kwargs` from scratch and drops pipeline-specific keys such as `padding="max_length"`.

## Fix / Workaround

Use shared `T5PaddedArchConfig` / `T5PaddedConfig` (`fastvideo/configs/models/encoders/t5.py`) so padding survives arch refresh. VACE and LongCat both import these symbols instead of duplicating arch subclasses.

## Prevention

When adding a model that needs non-default tokenizer kwargs, extend a dedicated arch config whose `__post_init__` re-applies the kwargs after `super().__post_init__()`, or centralize the contract in `t5.py`.
