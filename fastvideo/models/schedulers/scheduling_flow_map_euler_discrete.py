# SPDX-License-Identifier: Apache-2.0
"""Flow-map any-step Euler scheduler for AnyFlow.

The model predicts the *average* velocity ``u_θ(x_t, t, r)`` from time
``t`` back to time ``r``, so one Euler step is

    x_r = x_t - ((t - r) / num_train_timesteps) * u_θ(x_t, t, r)

regardless of how far apart ``t`` and ``r`` are. The scheduler also
provides the AnyFlow training-time helpers ``apply_shift`` (flow-matching
shift transform) and ``get_train_weight`` (per-timestep loss weight,
including ``beta08``).

Standalone — does not depend on diffusers' ConfigMixin/SchedulerMixin.
"""

from __future__ import annotations

from typing import Literal

import torch

from fastvideo.models.schedulers.base import BaseScheduler


WeightType = Literal["uniform", "gaussian", "beta08"]


class FlowMapEulerDiscreteScheduler(BaseScheduler):
    """Minimal flow-map scheduler.

    Parameters
    ----------
    num_train_timesteps:
        Discretization granularity for training. ``t`` is expressed in
        absolute units in ``[0, num_train_timesteps]``.
    shift:
        Flow-matching shift parameter (Wan video default: ``5.0``). Set
        to ``1.0`` for an identity shift.
    """

    order: int = 1

    def __init__(
        self,
        *,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
    ) -> None:
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.timesteps: torch.Tensor = torch.empty(0)
        self.sigmas: torch.Tensor = torch.empty(0)
        super().__init__()

    # ------------------------------------------------------------------
    # BaseScheduler abstract surface

    def set_shift(self, shift: float) -> None:
        self.shift = float(shift)

    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: int | None = None,
    ) -> torch.Tensor:
        # Flow-matching has no per-step input scaling; pass through.
        del timestep
        return sample

    # ------------------------------------------------------------------
    # Public helpers used by the AnyFlow pretrain method.

    def apply_shift(
        self,
        t: torch.Tensor,
        *,
        shift: float | None = None,
    ) -> torch.Tensor:
        """Apply the flow-matching shift: ``t' = s * t / (1 + (s - 1) * t)``.

        Operates in the normalized ``[0, 1]`` domain — callers should pass
        ``t / num_train_timesteps`` (or sample ``t`` directly from
        ``[0, 1]``).
        """
        s = self.shift if shift is None else float(shift)
        if s == 1.0:
            return t
        return s * t / (1.0 + (s - 1.0) * t)

    def get_train_weight(
        self,
        t: torch.Tensor,
        *,
        weight_type: WeightType = "beta08",
    ) -> torch.Tensor:
        """Per-timestep training weight, renormalized so the total weight
        mass equals ``num_train_timesteps`` (matching AnyFlow reference's
        ``scheduling_flowmap_euler_discrete.py``).

        ``beta08``: ``w(t) = t * sqrt(1 - t)`` (in normalized t-space).
        """
        # Auto-detect domain: if t was given in absolute units, normalize.
        t_f = t.float()
        max_val = t_f.max() if t_f.numel() > 0 else torch.tensor(0.0)
        if max_val > 1.0 + 1e-6:
            t_norm = t_f / self.num_train_timesteps
        else:
            t_norm = t_f
        t_norm = t_norm.clamp(min=0.0, max=1.0)

        if weight_type == "uniform":
            w = torch.ones_like(t_norm)
        elif weight_type == "gaussian":
            w = torch.exp(-0.5 * ((t_norm - 0.5) / 0.2) ** 2)
        elif weight_type == "beta08":
            w = t_norm.pow(1.0) * (1.0 - t_norm).clamp_min(0.0).pow(0.5)
        else:
            raise ValueError(f"Unknown weight_type: {weight_type!r}")

        denom = w.sum().clamp_min(1e-8)
        return w * (float(self.num_train_timesteps) / denom)

    # ------------------------------------------------------------------

    def set_timesteps(
        self,
        *,
        num_inference_steps: int,
        device: torch.device | str = "cpu",
        custom_timesteps: list[float] | torch.Tensor | None = None,
    ) -> None:
        """Build a descending timestep schedule ending at 0.

        With ``num_inference_steps=N`` the schedule has ``N + 1`` entries
        ``[T_max, ..., 0]`` so a rollout consumes ``N`` Euler steps.

        ``custom_timesteps`` overrides the linspace+shift schedule with
        a pinned list (in absolute train-timestep units), useful for the
        AnyFlow paper's hand-tuned ``[999, 937, 833, 624, 0]`` schedule.
        """
        if num_inference_steps <= 0:
            raise ValueError(
                "num_inference_steps must be positive, "
                f"got {num_inference_steps}")
        device = torch.device(device)

        if custom_timesteps is not None:
            ts = torch.as_tensor(
                custom_timesteps, dtype=torch.float32, device=device)
            if ts.ndim != 1:
                raise ValueError(
                    "custom_timesteps must be 1-D, got shape "
                    f"{tuple(ts.shape)}")
            if not torch.all(ts[:-1] >= ts[1:]):
                raise ValueError(
                    "custom_timesteps must be descending (largest first)")
        else:
            ts_norm = torch.linspace(
                1.0, 0.0, num_inference_steps + 1, device=device)
            ts_norm = self.apply_shift(ts_norm)
            ts = ts_norm * self.num_train_timesteps

        self.timesteps = ts
        self.sigmas = ts / self.num_train_timesteps

    def step(
        self,
        model_output: torch.Tensor,
        *,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        r_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """One Euler step from ``t`` to ``r``.

        ``model_output`` is the average-velocity prediction
        ``u_θ(x_t, t, r)``. Both ``timestep`` and ``r_timestep`` are in
        absolute train-timestep units (``[0, num_train_timesteps]``).
        """
        t = timestep.to(sample.device, dtype=sample.dtype)
        r = r_timestep.to(sample.device, dtype=sample.dtype)
        dt_norm = (t - r) / float(self.num_train_timesteps)
        # Broadcast dt over channel/spatial dims.
        view: list[int] = [-1] + [1] * (sample.ndim - 1)
        return sample - dt_norm.view(*view) * model_output

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Linear flow-matching interpolation: ``x_t = (1 - σ) * x_0 + σ * ε``,
        where ``σ = t / num_train_timesteps``.
        """
        sigma = (timestep.to(original_samples.device,
                             dtype=original_samples.dtype)
                 / float(self.num_train_timesteps))
        view: list[int] = [-1] + [1] * (original_samples.ndim - 1)
        sigma = sigma.view(*view)
        return (1.0 - sigma) * original_samples + sigma * noise
