---
date: 2026-04-22
experiment: daVinci-MagiHuman port (eval/davinci-port-test cold test)
category: porting
severity: important
---

# Timestep-free DiT requires a fully custom denoising stage

## What Happened
daVinci-MagiHuman has no timestep conditioning — no AdaLN, no time embedding,
no `t` argument to the DiT's forward pass. FastVideo's standard `DenoisingStage`
assumes timestep embeddings are injected into the DiT at every step. Using it
for a timestep-free model either crashes or silently passes `t=0` everywhere.

## Root Cause
`DenoisingStage` calls `dit(x, timestep=t, ...)` on every denoising step.
A timestep-free DiT's `forward()` signature doesn't accept `timestep` at all.
The stage also doesn't own the scheduler loop — it delegates to the base class
which assumes the standard flow.

## Fix / Workaround
Create a fully custom denoising stage that owns the entire scheduler loop:

```python
class DaVinciDenoisingStage(PipelineStage):
    def forward(self, batch, pipeline):
        scheduler = pipeline.scheduler
        for t in scheduler.timesteps:
            noise_pred = pipeline.dit(x=batch.latents, ...)  # no timestep arg
            batch.latents = scheduler.step(noise_pred, t, batch.latents).prev_sample
        return batch
```

The stage controls the loop, the scheduler step, and any model-specific logic
(e.g. dynamic CFG, audio fallback, coordinate assembly).

## Prevention
- During recon (Phase 0), check whether the official DiT `forward()` accepts
  a `timestep` / `t` argument. If not, flag it immediately — you will need a
  custom denoising stage.
- Add this to the recon checklist: "Is the model timestep-free? Search for
  `adaln`, `time_embed`, `timestep` in the model class. If absent → custom stage."
- Do NOT attempt to shoehorn a timestep-free model into `DenoisingStage` by
  passing a dummy `t` — it will appear to work but produce wrong outputs.