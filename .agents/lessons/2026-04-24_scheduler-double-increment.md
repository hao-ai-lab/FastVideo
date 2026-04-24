---
date: 2026-04-24
experiment: daVinci-MagiHuman inference port
category: scheduler
severity: critical
---

# Sharing one scheduler across multiple modalities exhausts sigmas in N/2 steps

## What Happened
`DaVinciDenoisingStage` denoises video and audio latents in the same loop.
Both called `self.scheduler.step()` once per iteration. Each call increments
`_step_index` by 1, so after 3 of 5 iterations `step_index` reached 5 and
`self.sigmas[step_index + 1]` = `self.sigmas[6]` was OOB (size 6 array).

Error: `IndexError: index 6 is out of bounds for dimension 0 with size 6`
at `scheduling_flow_match_euler_discrete.py:523`.

Confusingly the progress bar read "40%" (2/5 done) because the *first* call
in iteration i=2 (the video step) succeeded before the audio step hit OOB.

## Root Cause
`FlowMatchEulerDiscreteScheduler.step()` unconditionally increments
`self._step_index` at the end of every call. Calling it twice per loop
iteration burns through sigmas at 2× the intended rate.

## Fix
Capture `sigma`, `sigma_next`, and `dt` **before** the video `scheduler.step()`
call, then do the secondary modality's Euler update **manually** — bypassing
`scheduler.step()` entirely:

```python
# Snapshot sigma before video step increments step_index
if self.scheduler.step_index is None:
    self.scheduler._init_step_index(t)
s_idx = self.scheduler.step_index
_sigmas = self.scheduler.sigmas
_is_last = (s_idx + 1 >= len(_sigmas))
_sigma_next = (
    _sigmas[0].new_zeros(()) if _is_last
    else _sigmas[s_idx + 1])
_dt = _sigma_next - _sigmas[s_idx]

# Video step (increments step_index once — correct)
latents = self.scheduler.step(velocity, t, latents, ...)[0]

# Audio manual Euler (no scheduler.step call — no double-increment)
if not _is_last:
    audio_latents = (audio_latents + _dt * audio_velocity).to(dtype)
```

The non-stochastic Euler formula is `x_{t+1} = x_t + dt * v`, which is
exactly what `scheduler.step` computes internally for `stochastic_sampling=False`.

## Prevention
Any time a denoising loop steps multiple latent tensors (video + audio,
multiple resolution streams, etc.) with a **shared** scheduler instance,
only one of them should call `scheduler.step()`. All others should do the
Euler update manually using the pre-captured `dt`. Alternatively, give each
modality its own scheduler instance and call `set_timesteps` on both.
