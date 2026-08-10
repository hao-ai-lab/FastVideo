# SPDX-License-Identifier: Apache-2.0
"""On-device (MLX) DMD sampling for the FastWan runtime.

The hybrid proof-of-concept ran the FastWan DiT in MLX but bounced every
denoising step back through torch/NumPy to run the DMD scheduler math
(``MLX -> np.array -> torch (CPU) -> np.array -> MLX``). That host round-trip
forces a full device sync per step and defeats MLX's lazy graph execution.

This module mirrors the exact DMD arithmetic from
``fastvideo/models/utils.py::pred_noise_to_pred_video`` and
``FlowMatchEulerDiscreteScheduler.add_noise`` while keeping every large tensor
on the MLX device. The schedule lookup (``argmin`` over the ~1000-entry
training schedule) is done once on the host in NumPy: it is tiny, it is the
same value torch would compute, and it sidesteps the reduction-index quirk that
affects ``argmin`` on the Metal/MPS backends (see the CPU fallbacks in
``fastvideo/models/utils.py`` and ``scheduling_flow_match_euler_discrete.py``).

Because the DMD loop applies a single scalar timestep per step, ``sigma`` is a
scalar and the update is a plain elementwise affine combination — no
permute/flatten reshaping is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import mlx.core as mx


@dataclass(frozen=True)
class MLXDMDSchedule:
    """Host-side copy of a flow-match scheduler's ``(sigmas, timesteps)``.

    Holds the full training schedule so a DMD timestep (e.g. one of
    ``1000, 757, 522``) can be mapped to its flow-match ``sigma`` with the same
    nearest-timestep lookup the torch path uses.
    """

    sigmas: np.ndarray
    timesteps: np.ndarray

    @classmethod
    def from_torch_scheduler(cls, scheduler: Any) -> "MLXDMDSchedule":
        """Snapshot ``scheduler.sigmas`` / ``scheduler.timesteps`` to NumPy.

        Matches ``pred_noise_to_pred_video`` / ``add_noise``, which index the
        scheduler's *full* training schedule (not the per-inference subset).
        """
        sigmas = scheduler.sigmas.detach().to("cpu").double().numpy()
        timesteps = scheduler.timesteps.detach().to("cpu").double().numpy()
        return cls(sigmas=np.asarray(sigmas), timesteps=np.asarray(timesteps))

    def sigma_for(self, timestep: float) -> float:
        """Nearest-timestep ``sigma`` lookup, computed on the host."""
        idx = int(np.argmin(np.abs(self.timesteps - float(timestep))))
        return float(self.sigmas[idx])


def pred_noise_to_pred_video(
    pred_noise: "mx.array",
    noise_input_latent: "mx.array",
    sigma: float,
) -> "mx.array":
    """``x0`` prediction from a flow-match noise prediction (scalar ``sigma``).

    Equivalent to ``fastvideo/models/utils.py::pred_noise_to_pred_video`` for a
    single scalar timestep: ``pred_video = noise_input - sigma * pred_noise``.
    """
    return noise_input_latent - sigma * pred_noise


def add_noise(
    clean_latent: "mx.array",
    noise: "mx.array",
    sigma: float,
) -> "mx.array":
    """Flow-match forward noising, mirroring the scheduler's ``add_noise``.

    ``sample = (1 - sigma) * clean_latent + sigma * noise``.
    """
    return (1.0 - sigma) * clean_latent + sigma * noise


def dmd_step(
    *,
    latents: "mx.array",
    noise_input_latent: "mx.array",
    pred_noise: "mx.array",
    schedule: MLXDMDSchedule,
    timestep: float,
    next_timestep: float | None,
    noise: "mx.array | None" = None,
) -> "mx.array":
    """One on-device DMD update.

    Predicts the clean video from ``pred_noise`` at ``timestep`` and, when a
    ``next_timestep`` is given, re-noises it to that level (the intermediate
    DMD steps). The final step (``next_timestep is None``) returns the clean
    prediction directly.

    Args:
        latents: unused placeholder kept for call-site symmetry with the torch
            loop; the update depends only on ``noise_input_latent`` and
            ``pred_noise``.
        noise_input_latent: the noisy latent fed to the DiT this step.
        pred_noise: the DiT's velocity/noise prediction this step.
        schedule: host-side flow-match schedule.
        timestep: current DMD timestep.
        next_timestep: the DMD timestep to re-noise toward, or ``None`` on the
            last step.
        noise: fresh Gaussian noise (same shape as ``pred_noise``) for the
            re-noise; required when ``next_timestep`` is not ``None``.

    Returns:
        The latents to carry into the next step (or the final clean latents).
    """
    del latents  # symmetry with the torch loop; not needed for the math.
    sigma = schedule.sigma_for(timestep)
    pred_video = pred_noise_to_pred_video(pred_noise, noise_input_latent, sigma)
    if next_timestep is None:
        return pred_video
    if noise is None:
        raise ValueError("dmd_step requires `noise` when `next_timestep` is set (re-noise step).")
    sigma_next = schedule.sigma_for(next_timestep)
    return add_noise(pred_video, noise, sigma_next)
