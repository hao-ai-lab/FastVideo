# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The MiniMax authors and The HuggingFace Team.

from dataclasses import dataclass

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput


@dataclass
class MiniMaxH3SchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class MiniMaxH3Scheduler(SchedulerMixin, ConfigMixin):
    """Rectified-flow scheduler using MiniMax-H3's clean-time convention."""

    _compatibles: list[str] = []
    order = 1

    @register_to_config
    def __init__(self, shift: float = 12.0) -> None:
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}.")
        self.num_inference_steps: int | None = None
        self.sigmas: torch.Tensor | None = None
        self.timesteps: torch.Tensor | None = None
        self._shift = float(shift)
        self._step_index: int | None = None
        self._begin_index: int | None = None

    @property
    def shift(self) -> float:
        return self._shift

    @property
    def step_index(self) -> int | None:
        return self._step_index

    @property
    def begin_index(self) -> int | None:
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = begin_index

    def set_shift(self, shift: float) -> None:
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}.")
        self._shift = float(shift)

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        sigmas: list[float] | torch.Tensor | None = None,
    ) -> None:
        if sigmas is None:
            if num_inference_steps is None or num_inference_steps < 2:
                raise ValueError("`set_timesteps` requires explicit `sigmas` or "
                                 f"`num_inference_steps` >= 2, got {num_inference_steps}.")
            base = torch.linspace(1.0, 0.0, int(num_inference_steps), dtype=torch.float32)
            sigma_tensor = self._shift * base / (1 + (self._shift - 1) * base)
            sigma_tensor = torch.unique_consecutive(sigma_tensor)
        else:
            sigma_tensor = torch.as_tensor(sigmas, dtype=torch.float32).flatten().cpu()
            is_valid = (sigma_tensor.numel() >= 2 and bool((sigma_tensor[1:] < sigma_tensor[:-1]).all())
                        and sigma_tensor[-1].item() == 0.0)
            if not is_valid:
                raise ValueError("`sigmas` must hold at least two strictly decreasing values ending at 0.0.")

        self.sigmas = sigma_tensor.to(device=device)
        self.timesteps = (1.0 - sigma_tensor[:-1]).to(device=device)
        self.num_inference_steps = int(self.timesteps.numel())
        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep: float | torch.Tensor) -> int:
        if self.timesteps is None:
            raise ValueError("Call `set_timesteps` before looking up a timestep.")
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        indices = (self.timesteps == timestep).nonzero()
        if len(indices) == 0:
            raise ValueError("Passed `timestep` is not in `self.timesteps`.")
        return indices[0].item()

    def scale_model_input(self, sample: torch.Tensor, timestep: float | torch.Tensor | None = None) -> torch.Tensor:
        return sample

    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: float | torch.FloatTensor,
        noise: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, dtype=sample.dtype, device=sample.device)
        timestep = timestep.to(device=sample.device, dtype=sample.dtype)
        while timestep.ndim < sample.ndim:
            timestep = timestep.unsqueeze(-1)
        return timestep * sample + (1.0 - timestep) * noise

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: float | torch.FloatTensor,
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ) -> MiniMaxH3SchedulerOutput | tuple[torch.FloatTensor]:
        if isinstance(timestep, int) or (isinstance(timestep, torch.Tensor) and not timestep.is_floating_point()):
            raise ValueError("Pass a floating-point value from `scheduler.timesteps`, not an integer index.")
        if self.sigmas is None or self.timesteps is None:
            raise ValueError("Call `set_timesteps` before `step`.")
        if self._step_index is None:
            self._step_index = self.index_for_timestep(timestep) if self._begin_index is None else self._begin_index

        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, dtype=sample.dtype)
        # H3 deliberately derives x0's sigma from the transformer timestep, while
        # the Euler ratio below uses the stored grid. Keep the two float32 paths separate.
        sigma_from_timestep = 1 - timestep.to(device=sample.device, dtype=sample.dtype)
        while sigma_from_timestep.ndim < sample.ndim:
            sigma_from_timestep = sigma_from_timestep.unsqueeze(-1)
        denoised = sample + sigma_from_timestep * model_output

        compute_dtype = torch.float32 if sample.dtype in (torch.float16, torch.bfloat16) else sample.dtype
        sigma = self.sigmas[self._step_index].to(device=sample.device, dtype=compute_dtype)
        sigma_next = self.sigmas[self._step_index + 1].to(device=sample.device, dtype=compute_dtype)
        ratio = sigma_next / sigma
        prev_sample = ratio * sample.to(compute_dtype) + (1.0 - ratio) * denoised.to(compute_dtype)
        prev_sample = prev_sample.to(sample.dtype)
        self._step_index += 1

        if not return_dict:
            return (prev_sample, )
        return MiniMaxH3SchedulerOutput(prev_sample=prev_sample)


EntryClass = MiniMaxH3Scheduler
