---
date: 2026-04-22
experiment: daVinci-MagiHuman port (eval/davinci-port-test cold test)
category: porting
severity: moderate
---

# VAE latent normalization stats may not exist for non-standard z_dim

## What Happened
FastVideo's `WanVAEArchConfig` stores per-channel latent normalization stats
(`mean` and `std`, one value per latent channel). When porting daVinci, the
VAE uses `z_dim=48` (vs. Wan 2.1's `z_dim=16`). No published normalization
stats exist for 48 channels, so the config had to be stubbed with zeros/ones.

Stubbed stats silently produce unnormalized latents — the model may run without
errors but output quality will be degraded until real stats are calibrated.

## Root Cause
Normalization stats are dataset- and model-specific. They're computed by running
the VAE encoder over a representative dataset and computing per-channel mean/std.
When a new VAE variant ships without a published config file, these stats aren't
available.

## Fix / Workaround
Stub with zeros (mean) and ones (std) as placeholders, and flag explicitly:

```python
# TODO: calibrate real per-channel stats by running VAE encoder over representative dataset
latent_mean: list[float] = field(default_factory=lambda: [0.0] * 48)
latent_std: list[float] = field(default_factory=lambda: [1.0] * 48)
```

Add a warning log in the VAE config `__post_init__` if the stats are all
zeros/ones, so future runs surface the gap.

## Prevention
- During Phase 0 recon, check the official repo for a published config JSON
  with `latent_mean`/`latent_std` (or equivalent). Common locations:
  `config.json`, `vae_config.json`, `model_index.json`.
- If missing, note it as a known gap in the port checklist before Phase 1.
- Do not block Phase 1–3 on this — stub and flag. Real calibration needs GPU
  and representative data; it's a separate offline step.