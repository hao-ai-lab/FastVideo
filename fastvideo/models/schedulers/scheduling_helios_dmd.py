# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2025 The Helios Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Native scheduler for the distilled three-stage Helios pyramid."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput

from fastvideo.models.schedulers.base import BaseScheduler


@dataclass
class HeliosDMDSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor
    model_outputs: torch.FloatTensor | None = None
    last_sample: torch.FloatTensor | None = None
    this_order: int | None = None


class HeliosDMDScheduler(SchedulerMixin, ConfigMixin, BaseScheduler):
    """DMD flow scheduler used by `BestWishYsh/Helios-Distilled`."""

    _compatibles: list[Any] = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        stages: int = 3,
        stage_range: list[float] | None = None,
        gamma: float = 1 / 3,
        prediction_type: str = "flow_prediction",
        use_flow_sigmas: bool = True,
        use_dynamic_shifting: bool = False,
        time_shift_type: Literal["exponential", "linear"] = "linear",
        scheduler_type: str = "dmd",
        _diffusers_version: str | None = None,
        **kwargs,
    ) -> None:
        del scheduler_type, _diffusers_version, kwargs
        if stage_range is None:
            stage_range = [0, 1 / 3, 2 / 3, 1]
            self.register_to_config(stage_range=stage_range)
        self.num_train_timesteps = num_train_timesteps
        self.timestep_ratios: dict[int, tuple[float, float]] = {}
        self.timesteps_per_stage: dict[int, torch.Tensor] = {}
        self.sigmas_per_stage: dict[int, torch.Tensor] = {}
        self.start_sigmas: dict[int, float] = {}
        self.end_sigmas: dict[int, float] = {}
        self.ori_start_sigmas: dict[int, float] = {}

        self.init_sigmas_for_each_stage()
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()
        self.gamma = gamma
        self.last_sample = None
        self._step_index = None
        self._begin_index = None
        BaseScheduler.__init__(self)

    def init_sigmas(self) -> None:
        alphas = np.linspace(
            1,
            1 / self.config.num_train_timesteps,
            self.config.num_train_timesteps + 1,
        )
        sigmas = 1.0 - alphas
        sigmas = np.flip(self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas))[:-1].copy()
        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = (self.sigmas * self.config.num_train_timesteps).clone()
        self._step_index = None
        self._begin_index = None

    def init_sigmas_for_each_stage(self) -> None:
        self.init_sigmas()
        stage_distance = []
        training_steps = self.config.num_train_timesteps
        for stage_index in range(self.config.stages):
            start_index = max(int(self.config.stage_range[stage_index] * training_steps), 0)
            end_index = min(
                int(self.config.stage_range[stage_index + 1] * training_steps),
                training_steps,
            )
            start_sigma = self.sigmas[start_index].item()
            end_sigma = self.sigmas[end_index].item() if end_index < training_steps else 0.0
            self.ori_start_sigmas[stage_index] = start_sigma
            if stage_index != 0:
                original_sigma = 1 - start_sigma
                corrected_sigma = (1 / (math.sqrt(1 + 1 / self.config.gamma) * (1 - original_sigma) + original_sigma) *
                                   original_sigma)
                start_sigma = 1 - corrected_sigma
            stage_distance.append(start_sigma - end_sigma)
            self.start_sigmas[stage_index] = start_sigma
            self.end_sigmas[stage_index] = end_sigma

        total_distance = sum(stage_distance)
        for stage_index in range(self.config.stages):
            start_ratio = 0.0 if stage_index == 0 else sum(stage_distance[:stage_index]) / total_distance
            end_ratio = (0.9999999999999999 if stage_index == self.config.stages -
                         1 else sum(stage_distance[:stage_index + 1]) / total_distance)
            self.timestep_ratios[stage_index] = (start_ratio, end_ratio)

        for stage_index in range(self.config.stages):
            start_ratio, end_ratio = self.timestep_ratios[stage_index]
            timestep_max = min(self.timesteps[int(start_ratio * training_steps)], 999)
            timestep_min = self.timesteps[min(int(end_ratio * training_steps), training_steps - 1)]
            timesteps = np.linspace(timestep_max, timestep_min, training_steps + 1)
            self.timesteps_per_stage[stage_index] = (timesteps[:-1] if isinstance(timesteps, torch.Tensor) else
                                                     torch.from_numpy(timesteps[:-1]))
            stage_sigmas = np.linspace(0.999, 0, training_steps + 1)
            self.sigmas_per_stage[stage_index] = torch.from_numpy(stage_sigmas[:-1])

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = begin_index

    def set_shift(self, shift: float) -> None:
        self.register_to_config(shift=shift)
        self.timestep_ratios.clear()
        self.timesteps_per_stage.clear()
        self.sigmas_per_stage.clear()
        self.start_sigmas.clear()
        self.end_sigmas.clear()
        self.ori_start_sigmas.clear()
        self.init_sigmas_for_each_stage()

    def scale_model_input(self, sample: torch.Tensor, timestep: int | None = None) -> torch.Tensor:
        del timestep
        return sample

    def set_timesteps(
        self,
        num_inference_steps: int,
        stage_index: int | None = None,
        device: str | torch.device | None = None,
        sigmas: np.ndarray | None = None,
        mu: float | None = None,
        is_amplify_first_chunk: bool = False,
    ) -> None:
        num_inference_steps = num_inference_steps * 2 + 1 if is_amplify_first_chunk else num_inference_steps + 1
        self.num_inference_steps = num_inference_steps
        self.init_sigmas()

        if self.config.stages == 1:
            if sigmas is None:
                sigmas = np.linspace(
                    1,
                    1 / self.config.num_train_timesteps,
                    num_inference_steps + 1,
                )[:-1].astype(np.float32)
                if self.config.shift != 1.0:
                    if self.config.use_dynamic_shifting:
                        raise ValueError("Fixed shift and dynamic shifting cannot both be active")
                    sigmas = self.time_shift(self.config.shift, 1.0, sigmas)
            timesteps = (sigmas * self.config.num_train_timesteps).copy()
            sigma_tensor = torch.from_numpy(sigmas)
        else:
            if stage_index is None:
                raise ValueError("stage_index is required for multi-stage Helios")
            stage_timesteps = self.timesteps_per_stage[stage_index]
            timesteps = np.linspace(
                stage_timesteps[0].item(),
                stage_timesteps[-1].item(),
                num_inference_steps,
            )
            stage_sigmas = self.sigmas_per_stage[stage_index]
            ratios = np.linspace(
                stage_sigmas[0].item(),
                stage_sigmas[-1].item(),
                num_inference_steps,
            )
            sigma_tensor = torch.from_numpy(ratios)

        self.timesteps = torch.from_numpy(timesteps).to(device=device)
        self.sigmas = torch.cat([sigma_tensor, torch.zeros(1)]).to(device=device)
        self._step_index = None
        self.reset_scheduler_history()
        self.timesteps = self.timesteps[:-1]
        self.sigmas = torch.cat([self.sigmas[:-2], self.sigmas[-1:]])

        if self.config.use_dynamic_shifting:
            if self.config.shift != 1.0:
                raise ValueError("Dynamic shifting requires shift=1.0")
            if mu is None:
                raise ValueError("Dynamic shifting requires mu")
            self.sigmas = self.time_shift(mu, 1.0, self.sigmas)
            if self.config.stages == 1:
                self.timesteps = self.sigmas[:-1] * self.config.num_train_timesteps
            else:
                assert stage_index is not None
                stage_timesteps = self.timesteps_per_stage[stage_index]
                self.timesteps = stage_timesteps.min() + self.sigmas[:-1] * (stage_timesteps.max() -
                                                                             stage_timesteps.min())

    def time_shift(self, mu: float, sigma: float, timesteps):
        if self.config.time_shift_type == "exponential":
            return math.exp(mu) / (math.exp(mu) + (1 / timesteps - 1)**sigma)
        if self.config.time_shift_type == "linear":
            return mu / (mu + (1 / timesteps - 1)**sigma)
        raise ValueError(f"Unknown time_shift_type: {self.config.time_shift_type}")

    @staticmethod
    def add_noise(
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
        sigmas: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        sigmas = sigmas.to(noise.device)
        timesteps = timesteps.to(noise.device)
        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = sigmas[timestep_id].reshape(-1, 1, 1, 1, 1)
        return ((1 - sigma) * original_samples + sigma * noise).type_as(noise)

    @staticmethod
    def convert_flow_pred_to_x0(
        flow_pred: torch.Tensor,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        sigmas: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        original_dtype = flow_pred.dtype
        device = flow_pred.device
        flow_pred, sample, sigmas, timesteps = (value.double().to(device)
                                                for value in (flow_pred, sample, sigmas, timesteps))
        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = sigmas[timestep_id].reshape(-1, 1, 1, 1, 1)
        return (sample - sigma * flow_pred).to(original_dtype)

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: float | torch.FloatTensor | None = None,
        sample: torch.FloatTensor | None = None,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
        cur_sampling_step: int = 0,
        dmd_noisy_tensor: torch.FloatTensor | None = None,
        dmd_sigmas: torch.FloatTensor | None = None,
        dmd_timesteps: torch.FloatTensor | None = None,
        all_timesteps: torch.FloatTensor | None = None,
    ) -> HeliosDMDSchedulerOutput | tuple[torch.Tensor]:
        del generator
        if (timestep is None or sample is None or dmd_noisy_tensor is None or dmd_sigmas is None
                or dmd_timesteps is None or all_timesteps is None):
            raise ValueError("Helios DMD step requires all stage-local tensors")
        predicted_x0 = self.convert_flow_pred_to_x0(
            model_output,
            sample,
            torch.full(
                (model_output.shape[0], ),
                timestep,
                dtype=torch.long,
                device=model_output.device,
            ),
            dmd_sigmas,
            dmd_timesteps,
        )
        if cur_sampling_step < len(all_timesteps) - 1:
            prev_sample = self.add_noise(
                predicted_x0,
                dmd_noisy_tensor,
                torch.full(
                    (model_output.shape[0], ),
                    all_timesteps[cur_sampling_step + 1],
                    dtype=torch.long,
                    device=model_output.device,
                ),
                dmd_sigmas,
                dmd_timesteps,
            )
        else:
            prev_sample = predicted_x0
        if not return_dict:
            return (prev_sample, )
        return HeliosDMDSchedulerOutput(prev_sample=prev_sample)

    def reset_scheduler_history(self) -> None:
        self._step_index = None
        self._begin_index = None

    def __len__(self) -> int:
        return self.config.num_train_timesteps


EntryClass = HeliosDMDScheduler
