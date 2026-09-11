# SPDX-License-Identifier: Apache-2.0
"""Four-step rectified-flow ODE scheduler for the Cosmos Predict2.5 DFD student."""

from dataclasses import dataclass
from typing import Any

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput

from fastvideo.models.schedulers.base import BaseScheduler


@dataclass
class Cosmos25DFDSchedulerOutput(BaseOutput):
    prev_sample: torch.Tensor
    pred_original_sample: torch.Tensor


class Cosmos25DFDScheduler(SchedulerMixin, ConfigMixin, BaseScheduler):
    """Run the fixed four-step ODE trajectory used by the public DFD checkpoint."""

    _compatibles: list[Any] = []
    order = 1
    _SAMPLE_TIMES = (0.999, 0.937, 0.833, 0.624, 0.0)

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        sigma_data: float = 1.0,
    ) -> None:
        if sigma_data <= 0:
            raise ValueError(f"sigma_data must be positive, got {sigma_data}")
        self.num_train_timesteps = num_train_timesteps
        self.sigma_data = float(sigma_data)
        self.timesteps = torch.empty(0, dtype=torch.float64)
        self.sigmas = torch.empty(0, dtype=torch.float64)
        self._step_index: int | None = None
        self._begin_index: int | None = None
        BaseScheduler.__init__(self)

    @property
    def init_noise_sigma(self) -> float:
        return self._SAMPLE_TIMES[0]

    @property
    def step_index(self) -> int | None:
        return self._step_index

    @property
    def begin_index(self) -> int | None:
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = begin_index

    def set_shift(self, shift: float) -> None:
        """The checkpoint's learned timestep list must not be shifted."""
        del shift

    def set_timesteps(
        self,
        num_inference_steps: int = 4,
        device: str | torch.device | None = None,
    ) -> None:
        if num_inference_steps != 4:
            raise ValueError(f"Cosmos Predict2.5 DFD requires exactly 4 inference steps; got {num_inference_steps}")
        schedule = torch.tensor(self._SAMPLE_TIMES, dtype=torch.float64, device=device)
        self.num_inference_steps = num_inference_steps
        self.timesteps = schedule[:-1]
        self.sigmas = schedule
        self._step_index = None
        self._begin_index = None

    def _init_step_index(self, timestep: torch.Tensor | float) -> None:
        if self._begin_index is not None:
            self._step_index = self._begin_index
            return
        value = torch.as_tensor(timestep, dtype=self.timesteps.dtype, device=self.timesteps.device)
        matches = torch.isclose(self.timesteps, value)
        if not bool(matches.any()):
            raise ValueError(f"Timestep {float(value)} is not in the configured DFD schedule")
        self._step_index = int(matches.nonzero()[0].item())

    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        del timestep
        return sample

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> Cosmos25DFDSchedulerOutput | tuple[torch.Tensor, ...]:
        del generator
        if self._step_index is None:
            self._init_step_index(timestep)
        assert self._step_index is not None
        if self._step_index >= len(self.timesteps):
            raise IndexError("All configured Cosmos Predict2.5 DFD steps have already run")

        sigma = self.sigmas[self._step_index].to(device=sample.device, dtype=torch.float64)
        sigma_next = self.sigmas[self._step_index + 1].to(device=sample.device, dtype=torch.float64)
        sample_f64 = sample.to(torch.float64)
        model_output_f64 = model_output.to(torch.float64)

        pred_original_sample = sample_f64 - sigma * model_output_f64
        prev_sample = sample_f64 + (sigma_next - sigma) * model_output_f64

        self._step_index += 1
        prev_sample = prev_sample.to(sample.dtype)
        pred_original_sample = pred_original_sample.to(sample.dtype)
        if not return_dict:
            return (prev_sample,)
        return Cosmos25DFDSchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=pred_original_sample,
        )

    def scale_noise(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del sample, timestep
        if noise is None:
            raise ValueError("noise must be provided")
        return self.init_noise_sigma * noise

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        timesteps = timesteps.to(device=original_samples.device, dtype=original_samples.dtype)
        while timesteps.ndim < original_samples.ndim:
            timesteps = timesteps.unsqueeze(-1)
        return (1.0 - timesteps) * original_samples + timesteps * noise

    def __len__(self) -> int:
        return self.config.num_train_timesteps


EntryClass = Cosmos25DFDScheduler
