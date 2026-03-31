# SPDX-License-Identifier: Apache-2.0
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput
import torch

from fastvideo.models.schedulers.base import BaseScheduler


class DiffusionForcingSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class DiffusionForcingScheduler(BaseScheduler, ConfigMixin, SchedulerMixin):

    config_name = "scheduler_config.json"
    order = 1

    @register_to_config
    def __init__(
        self,
        num_inference_steps: int = 100,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.0,
        extra_one_step: bool = True,
        training: bool = False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.extra_one_step = extra_one_step
        self.set_timesteps(num_inference_steps, training=training)

    def _apply_shift(self, sigmas: torch.Tensor) -> torch.Tensor:
        """Apply time-shift to sigmas, matching FlowMatchEulerDiscreteScheduler."""
        if self.shift != 1.0:
            sigmas = (self.shift * sigmas
                      / (1 + (self.shift - 1) * sigmas))
        return sigmas

    def _lookup_sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        """Map timestep values to sigmas via table lookup."""
        timestep = timestep.to(torch.float32)
        self.timesteps = self.timesteps.to(timestep.device)
        self.sigmas = self.sigmas.to(timestep.device)
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(0)
             - timestep.unsqueeze(1)).abs(),
            dim=1,
        )
        return self.sigmas[timestep_id]

    def set_timesteps(
        self,
        num_inference_steps: int = 100,
        denoising_strength: float = 1.0,
        training: bool = False,
        return_dict: bool = False,
        **kwargs,
    ):
        if training:
            # Linear sigmas from 1.0 to 0.0 (inclusive),
            # then apply the same shift used by the student model.
            raw_sigmas = torch.arange(
                self.num_train_timesteps, -1, -1,
                dtype=torch.float32,
            ) / float(self.num_train_timesteps)
            sigmas = self._apply_shift(raw_sigmas)
            self.sigmas = sigmas
            self.timesteps = (
                sigmas * float(self.num_train_timesteps)
            ).to(torch.float32)
            self.linear_timesteps_weights = torch.ones_like(
                self.timesteps)
            return

        sigma_start = self.sigma_min + (
            self.sigma_max - self.sigma_min
        ) * denoising_strength
        if self.extra_one_step:
            sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps + 1
            )[:-1]
        else:
            sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps
            )
        sigmas = self._apply_shift(sigmas)
        self.sigmas = sigmas.to(torch.float32)
        self.timesteps = (
            self.sigmas * float(self.num_train_timesteps)
        ).to(torch.float32)

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: torch.FloatTensor,
        sample: torch.FloatTensor,
        to_final: bool = False,
        return_dict: bool = False,
        **kwargs,
    ):
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        elif timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)

        timestep = timestep.to(model_output.device, dtype=torch.float32)
        self.timesteps = self.timesteps.to(model_output.device)
        self.sigmas = self.sigmas.to(model_output.device)

        sigma = self._lookup_sigma(timestep).reshape(-1, 1, 1, 1)
        if to_final:
            sigma_next = torch.zeros_like(sigma)
        else:
            timestep_id = torch.argmin(
                (self.timesteps.unsqueeze(0)
                 - timestep.unsqueeze(1)).abs(),
                dim=1,
            )
            sigma_next = torch.zeros_like(sigma)
            valid = timestep_id + 1 < len(self.timesteps)
            if valid.any():
                sigma_next[valid] = self.sigmas[
                    timestep_id[valid] + 1
                ].reshape(-1, 1, 1, 1)

        prev_sample = sample + model_output * (sigma_next - sigma)
        if isinstance(prev_sample, torch.Tensor | float) and not return_dict:
            return (prev_sample,)
        return DiffusionForcingSchedulerOutput(prev_sample=prev_sample)

    @staticmethod
    def calculate_alpha_beta_high(sigma, sigma_bound):
        alpha = (1 - sigma) / (1 - sigma_bound)
        beta = torch.sqrt(sigma**2 - (alpha * sigma_bound) ** 2)
        return alpha, beta

    def add_noise(self, original_samples, noise, timestep):
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        sigma = self._lookup_sigma(timestep).to(noise.device).reshape(
            -1, 1, 1, 1
        )
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

    def add_noise_high(
        self, original_samples, noise, timestep, boundary_timestep
    ):
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        if boundary_timestep.ndim == 2:
            boundary_timestep = boundary_timestep.flatten(0, 1)
        sigma = self._lookup_sigma(timestep).to(noise.device).reshape(
            -1, 1, 1, 1
        )
        sigma_boundary = self._lookup_sigma(
            boundary_timestep
        ).to(noise.device).reshape(-1, 1, 1, 1)
        alpha, beta = self.calculate_alpha_beta_high(sigma, sigma_boundary)
        sample = alpha * original_samples + beta * noise
        return sample.type_as(noise)

    def sigma_from_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        """Public API used by ``pred_noise_to_pred_video``."""
        return self._lookup_sigma(timestep)

    def training_target(self, sample, noise, timestep):
        return noise - sample

    def training_weight(self, timestep):
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        return torch.ones_like(timestep, dtype=torch.float32)

    def scale_model_input(
        self, sample: torch.Tensor, timestep: int | None = None
    ) -> torch.Tensor:
        return sample

    def set_shift(self, shift: float) -> None:
        self.shift = shift


EntryClass = DiffusionForcingScheduler
