"""RCM (Reparameterized / recurrent Consistency Model) sampler helpers — TurboWan's few-step distilled
schedule + SDE consistency step.

Faithful CPU-testable port of ``fastvideo/models/schedulers/scheduling_rcm.py:RCMScheduler`` (the
TurboDiffusion rCM sampler, arXiv:2512.16093). The math has two pieces that this module mirrors exactly:

  1. ``build_rcm_sigmas`` — the TrigFlow -> RectifiedFlow timestep schedule. Starting from
     ``[atan(sigma_max), *mid_timesteps, 0]`` in TrigFlow angle space, each entry is converted with
     ``t = sin(t) / (cos(t) + sin(t))`` to a RectifiedFlow sigma in ``[0, 1]``. ``mid_timesteps`` defaults
     to ``[1.5, 1.4, 1.0]`` (the scheduler's visual-quality default), sliced to ``num_steps - 1`` so the
     final schedule has ``num_steps + 1`` points (the appended terminal ``0``). This matches the loop
     convention everywhere else in v2 (``i`` ranges over ``len(sigmas) - 1`` pairwise intervals).

  2. ``rcm_step`` — the rCM SDE consistency update. The model predicts a velocity ``v_pred`` at the
     CURRENT (raw) sigma ``t_cur``; the scheduler reconstructs the clean estimate
     ``x_denoised = x - t_cur * v_pred`` and re-noises it to the NEXT sigma with FRESH Gaussian noise:

         x_next = (1 - t_next) * x_denoised + t_next * noise

     This is NOT a deterministic flow-match Euler step — it is a stochastic consistency step that
     injects fresh noise every step (which is why TurboWan reaches quality in 1-4 steps). The fresh
     host RNG is exactly why the loop must run this on the eager path (like the FlowGRPO SDE rollout),
     never inside a captured CUDA graph.

The real scheduler scales the model-input timestep by 1000 (``timesteps = sigmas * 1000``) — the loop
passes that scaled timestep to the DiT; the step formula here uses the RAW sigmas. ``init_noise_sigma``
is ``sigmas[0]`` (the loop scales the initial latent by it, matching ``scale_noise``).
"""
from __future__ import annotations

import math

import numpy as np

# The scheduler's quality-tuned intermediate TrigFlow timesteps (used for 2-, 3- and 4-step schedules).
RCM_MID_TIMESTEPS = (1.5, 1.4, 1.0)


def build_rcm_sigmas(num_steps: int,
                     sigma_max: float = 80.0,
                     mid_timesteps: tuple[float, ...] = RCM_MID_TIMESTEPS) -> np.ndarray:
    """RCM raw-sigma schedule (``num_steps + 1`` points, terminal ``0``) in RectifiedFlow space.

    TrigFlow angles ``[atan(sigma_max), *mid[:num_steps-1], 0]`` -> RectifiedFlow sigmas via
    ``sin/(cos+sin)``. ``sigma_max`` is 80 for TurboWan T2V and 200 for the I2V MoE variant.
    """
    n = max(1, int(num_steps))
    mid = list(mid_timesteps[:max(0, n - 1)])
    angles = np.asarray([math.atan(float(sigma_max)), *mid, 0.0], dtype=np.float64)
    return (np.sin(angles) / (np.cos(angles) + np.sin(angles))).astype(np.float64)


def rcm_step(x_t: np.ndarray, velocity: np.ndarray, sigma_t: float, sigma_next: float, noise: np.ndarray) -> np.ndarray:
    """One rCM SDE consistency step: ``x_next = (1 - t_next)*(x_t - t_cur*v) + t_next*noise``.

    ``noise`` is fresh Gaussian noise of ``x_t``'s shape (drawn by the loop's per-request RNG). On the
    final interval ``sigma_next`` is ``0`` so the term collapses to the clean denoised estimate.
    """
    x = np.asarray(x_t, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    t_cur, t_next = float(sigma_t), float(sigma_next)
    x_denoised = x - t_cur * v
    x_next = (1.0 - t_next) * x_denoised + t_next * np.asarray(noise, dtype=np.float64)
    return x_next.astype("float32")
