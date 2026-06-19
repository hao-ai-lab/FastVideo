"""RCM (Reparameterized Consistency Model) sampler helpers — TurboWan's few-step schedule + SDE step.

Faithful CPU-testable port of ``fastvideo/models/schedulers/scheduling_rcm.py:RCMScheduler`` (TurboDiffusion
rCM sampler, arXiv:2512.16093). Two pieces mirrored exactly:

  1. ``build_rcm_sigmas`` — the TrigFlow -> RectifiedFlow timestep schedule. From the TrigFlow angles
     ``[atan(sigma_max), *mid_timesteps, 0]``, each entry is converted with ``t = sin(t)/(cos(t)+sin(t))`` to a
     RectifiedFlow sigma in ``[0, 1]``. ``mid_timesteps`` defaults to ``[1.5, 1.4, 1.0]`` (the scheduler's
     quality default), sliced to ``num_steps - 1`` so the schedule has ``num_steps + 1`` points (terminal
     ``0``), matching the v2 loop convention (``i`` ranges over ``len(sigmas) - 1`` intervals).

  2. ``rcm_step`` — the rCM SDE consistency update. The model predicts a velocity ``v_pred`` at the current
     (raw) sigma ``t_cur``; the scheduler reconstructs the clean estimate ``x_denoised = x - t_cur * v_pred``
     and re-noises it to the next sigma with fresh Gaussian noise:

         x_next = (1 - t_next) * x_denoised + t_next * noise

     This stochastic consistency step (not a deterministic Euler step) injects fresh noise every step, which
     is why TurboWan reaches quality in 1-4 steps — and why the loop must run it on the eager path (like the
     FlowGRPO SDE rollout), never inside a captured CUDA graph.

The real scheduler scales the model-input timestep by 1000 (``timesteps = sigmas * 1000``); the loop passes
that scaled timestep to the DiT while the step formula here uses the raw sigmas. ``init_noise_sigma`` is
``sigmas[0]`` (the loop scales the initial latent by it, matching ``scale_noise``).
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
