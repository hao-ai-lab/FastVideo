# SPDX-License-Identifier: Apache-2.0
"""TrigFlow sampler for the distilled Cosmos Predict2.5 checkpoint."""

import math
from dataclasses import dataclass
from typing import Any

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput

from fastvideo.models.schedulers.base import BaseScheduler


@dataclass
class Cosmos25DistilledSchedulerOutput(BaseOutput):
    """Output of one distilled Cosmos Predict2.5 sampling step."""

    prev_sample: torch.Tensor
    pred_original_sample: torch.Tensor


class Cosmos25DistilledScheduler(SchedulerMixin, ConfigMixin, BaseScheduler):
    """Official four-step TrigFlow/x0 sampler for Cosmos Predict2.5.

    The distilled student predicts the rectified-flow network output. The
    scheduler applies the student's TrigFlow preconditioning, reconstructs x0,
    and re-noises it with the *initial* noise for the next student evaluation.
    This fixed-noise update is intentionally different from stochastic rCM
    sampling.
    """

    _compatibles: list[Any] = []
    order = 1
    _OFFICIAL_SAMPLING_TIMES = (
        math.pi / 2,
        math.atan(15),
        math.atan(5),
        math.atan(5 / 3),
    )

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
        self.trigflow_timesteps = torch.empty(0, dtype=torch.float64)
        self.sigmas = torch.empty(0, dtype=torch.float64)
        self._step_index: int | None = None
        self._begin_index: int | None = None
        self._initial_noise: torch.Tensor | None = None
        BaseScheduler.__init__(self)

    @property
    def init_noise_sigma(self) -> float:
        return 1.0

    @property
    def step_index(self) -> int | None:
        return self._step_index

    @property
    def begin_index(self) -> int | None:
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = begin_index

    def set_shift(self, shift: float) -> None:
        """The checkpoint's fixed distilled schedule does not use flow shift."""

    def set_timesteps(
        self,
        num_inference_steps: int = 4,
        device: str | torch.device | None = None,
    ) -> None:
        if not 1 <= num_inference_steps <= len(self._OFFICIAL_SAMPLING_TIMES):
            raise ValueError(f"Cosmos Predict2.5 distilled sampling supports 1 to 4 steps; got {num_inference_steps}")

        trigflow_timesteps = torch.tensor(
            self._OFFICIAL_SAMPLING_TIMES[:num_inference_steps],
            dtype=torch.float64,
            device=device,
        )
        _, _, _, model_timesteps = self._scalings(trigflow_timesteps)

        self.num_inference_steps = num_inference_steps
        self.trigflow_timesteps = trigflow_timesteps
        # The Cosmos DiT consumes c_noise, not the TrigFlow angle.
        self.timesteps = model_timesteps
        self.sigmas = torch.tan(trigflow_timesteps) * self.sigma_data
        self._step_index = None
        self._initial_noise = None

    def _scalings(
        self,
        trigflow_timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = trigflow_timestep.dtype
        timestep = trigflow_timestep.to(torch.float64)
        denominator = torch.cos(timestep) + self.sigma_data * torch.sin(timestep)
        c_skip = self.sigma_data / denominator
        c_out = -self.sigma_data * torch.sin(timestep) / denominator
        c_in = self.sigma_data / denominator
        c_noise = self.sigma_data * torch.sin(timestep) / denominator
        return (
            c_skip.to(dtype),
            c_out.to(dtype),
            c_in.to(dtype),
            c_noise.to(dtype),
        )

    def _init_step_index(self) -> None:
        self._step_index = self._begin_index if self._begin_index is not None else 0

    def _current_scalings(
        self,
        sample: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._step_index is None:
            self._init_step_index()
        assert self._step_index is not None
        if self._step_index >= len(self.trigflow_timesteps):
            raise IndexError("All configured Cosmos Predict2.5 distilled steps have already run")
        timestep = self.trigflow_timesteps[self._step_index].to(device=sample.device)
        return self._scalings(timestep)

    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        del timestep
        _, _, c_in, _ = self._current_scalings(sample)
        return sample * c_in

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> Cosmos25DistilledSchedulerOutput | tuple[torch.Tensor, ...]:
        del timestep, generator
        c_skip, c_out, _, _ = self._current_scalings(sample)
        assert self._step_index is not None

        if self._initial_noise is None:
            # Official inference keeps init_noise in FP32 while carrying the
            # evolving sample in FP64.
            self._initial_noise = sample.detach().to(torch.float32).clone()

        pred_original_sample = c_skip * sample + c_out * model_output
        next_index = self._step_index + 1
        if next_index < len(self.trigflow_timesteps):
            next_timestep = self.trigflow_timesteps[next_index].to(
                device=sample.device,
                dtype=torch.float64,
            )
            prev_sample = (
                torch.cos(next_timestep) * pred_original_sample / self.sigma_data
                + torch.sin(next_timestep) * self._initial_noise
            )
        else:
            prev_sample = pred_original_sample

        self._step_index = next_index
        if not return_dict:
            return (prev_sample,)
        return Cosmos25DistilledSchedulerOutput(
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
        return noise

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("Cosmos25DistilledScheduler is an inference-only x0 sampler")

    def __len__(self) -> int:
        return self.config.num_train_timesteps


EntryClass = Cosmos25DistilledScheduler
