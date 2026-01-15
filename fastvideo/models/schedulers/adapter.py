# SPDX-License-Identifier: Apache-2.0
"""
Scheduler adapter interfaces for unified denoising.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch


class SchedulerAdapter(Protocol):
    def scale_model_input(self, latents: torch.Tensor,
                          t: torch.Tensor) -> torch.Tensor:
        ...

    def step(self, noise_pred: torch.Tensor, t: torch.Tensor,
             latents: torch.Tensor, **kwargs: Any) -> Any:
        ...

    def add_noise(self, latents: torch.Tensor, noise: torch.Tensor,
                  t: torch.Tensor) -> torch.Tensor:
        ...

    def set_timesteps(self, num_steps: int, device: torch.device | None = None,
                      **kwargs: Any) -> Any:
        ...


class DefaultSchedulerAdapter:
    def __init__(self, scheduler: Any) -> None:
        self.scheduler = scheduler

    def scale_model_input(self, latents: torch.Tensor,
                          t: torch.Tensor) -> torch.Tensor:
        return self.scheduler.scale_model_input(latents, t)

    def step(self, noise_pred: torch.Tensor, t: torch.Tensor,
             latents: torch.Tensor, **kwargs: Any) -> Any:
        return self.scheduler.step(noise_pred, t, latents, **kwargs)

    def add_noise(self, latents: torch.Tensor, noise: torch.Tensor,
                  t: torch.Tensor) -> torch.Tensor:
        return self.scheduler.add_noise(latents, noise, t)

    def set_timesteps(self, num_steps: int, device: torch.device | None = None,
                      **kwargs: Any) -> Any:
        return self.scheduler.set_timesteps(num_steps, device=device, **kwargs)
