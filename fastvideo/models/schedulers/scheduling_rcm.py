# SPDX-License-Identifier: Apache-2.0
# rCM (recurrent Consistency Model) Scheduler for TurboDiffusion support
#
# This scheduler implements the rCM sampling method from TurboDiffusion,
# enabling 1-4 step video generation with distilled checkpoints.
#
# Reference:
#   TurboDiffusion: Accelerating Video Diffusion Models by 100-200 Times
#   https://arxiv.org/pdf/2512.16093

import math
from dataclasses import dataclass
from typing import Any

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput

from fastvideo.logger import init_logger
from fastvideo.models.schedulers.base import BaseScheduler

logger = init_logger(__name__)


@dataclass
class RCMSchedulerOutput(BaseOutput):
    """
    Output class for the RCM scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, ...)`):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used
            as next model input in the denoising loop.
    """

    prev_sample: torch.FloatTensor


class RCMScheduler(SchedulerMixin, ConfigMixin, BaseScheduler):
    """
    rCM (recurrent Consistency Model) scheduler for TurboDiffusion.

    This scheduler implements the rCM sampling method which enables 1-4 step
    video generation using distilled checkpoints. It uses:
    1. TrigFlow → RectifiedFlow timestep conversion
    2. SDE sampling formula: x = (1 - t_next) * (x - t_cur * v_pred) + t_next * noise

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps used to train the model.
        sigma_max (`float`, defaults to 80.0):
            The initial sigma value for rCM sampling. Controls the noise level
            at the start of sampling.
        mid_timesteps (`list[float]`, *optional*):
            Custom intermediate timesteps. If None, uses optimized defaults
            [1.5, 1.4, 1.0] for better visual quality.
    """

    _compatibles: list[Any] = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        sigma_max: float = 80.0,
        mid_timesteps: list[float] | None = None,
    ):
        # Default mid timesteps optimized for visual quality
        if mid_timesteps is None:
            mid_timesteps = [1.5, 1.4, 1.0]
        self._mid_timesteps = mid_timesteps
        
        self.num_train_timesteps = num_train_timesteps
        self.sigma_max = sigma_max
        
        # Initialize with default timesteps (will be set properly via set_timesteps)
        self.timesteps = torch.tensor([1.0, 0.0], dtype=torch.float64)
        self.sigmas = self.timesteps.clone()
        
        self._step_index: int | None = None
        self._begin_index: int | None = None
        
        BaseScheduler.__init__(self)

    @property
    def step_index(self) -> int | None:
        """
        The index counter for current timestep. Increases by 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self) -> int | None:
        """
        The index for the first timestep.
        """
        return self._begin_index

    @property
    def init_noise_sigma(self) -> float:
        """
        Initial noise sigma for scaling latents.
        
        In rCM, initial noise is scaled by the first sigma value:
            x_0 = noise * sigmas[0]
        
        This property is used by LatentPreparationStage to scale initial latents.
        """
        return float(self.sigmas[0])

    def set_begin_index(self, begin_index: int = 0) -> None:
        """
        Sets the begin index for the scheduler.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def set_shift(self, shift: float) -> None:
        """
        rCM doesn't use shift parameter, but required by BaseScheduler.
        """
        pass

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: str | torch.device | None = None,
        sigma_max: float | None = None,
    ) -> None:
        """
        Sets the discrete timesteps used for the rCM sampling process.

        The timesteps are computed using TrigFlow → RectifiedFlow conversion:
        1. Start with atan(sigma_max) and intermediate values
        2. Convert via: t = sin(t) / (cos(t) + sin(t))

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps (1-4 for rCM).
            device (`str` or `torch.device`, *optional*):
                The device to move timesteps to.
            sigma_max (`float`, *optional*):
                Override the initial sigma value.
        """
        if num_inference_steps < 1 or num_inference_steps > 4:
            logger.warning(
                "rCM is optimized for 1-4 steps, got %d steps. "
                "Performance may be suboptimal.", num_inference_steps
            )

        self.num_inference_steps = num_inference_steps
        
        if sigma_max is not None:
            self.sigma_max = sigma_max

        # Build timestep schedule
        mid_t = self._mid_timesteps[:num_inference_steps - 1]

        # TrigFlow timesteps: [atan(sigma_max), mid_t..., 0]
        t_steps = torch.tensor(
            [math.atan(self.sigma_max), *mid_t, 0],
            dtype=torch.float64,
            device=device,
        )

        # Convert TrigFlow → RectifiedFlow: t = sin(t) / (cos(t) + sin(t))
        t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

        # Store raw sigmas for use in step() formula
        self.sigmas = t_steps.clone()
        
        # Scale timesteps by 1000 for model input (as per TurboDiffusion)
        self.timesteps = t_steps * 1000
        
        self._step_index = None
        self._begin_index = None

        logger.debug("rCM timesteps (scaled): %s", self.timesteps.tolist())
        logger.debug("rCM sigmas (raw): %s", self.sigmas.tolist())

    def _init_step_index(self, timestep: torch.FloatTensor | None = None) -> None:
        """Initialize step index at the beginning of sampling."""
        if self._begin_index is None:
            self._step_index = 0
        else:
            self._step_index = self._begin_index

    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: int | None = None,
    ) -> torch.Tensor:
        """
        rCM doesn't scale model input, returns sample as-is.
        """
        return sample

    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: torch.FloatTensor | None = None,
        noise: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        """
        Scale initial noise for rCM sampling.
        
        In rCM, initial noise is scaled by the first timestep (raw sigma):
            x_0 = noise * t_steps[0]

        Args:
            sample: Not used (for API compatibility)
            timestep: Not used (for API compatibility)
            noise: The noise tensor to scale

        Returns:
            Scaled noise tensor ready for sampling
        """
        if noise is None:
            raise ValueError("noise must be provided for rCM scale_noise")
        
        # Use raw sigma (not scaled timestep) for initial noise scaling
        t_initial = self.sigmas[0]
        return noise.to(torch.float64) * t_initial

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int | torch.Tensor,
        sample: torch.FloatTensor,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> RCMSchedulerOutput | tuple[torch.FloatTensor, ...]:
        """
        Predict the sample from the previous timestep using rCM update rule.

        The rCM update formula is:
            x_{t+1} = (1 - t_next) * (x_t - t_cur * v_pred) + t_next * noise

        Args:
            model_output (`torch.FloatTensor`):
                The velocity prediction from the model (v_pred).
            timestep (`int` or `torch.Tensor`):
                The current timestep index (not the actual timestep value).
                For rCM, this should be the index into self.timesteps.
            sample (`torch.FloatTensor`):
                Current sample x_t.
            generator (`torch.Generator`, *optional*):
                Random number generator for noise.
            return_dict (`bool`):
                Whether to return RCMSchedulerOutput or tuple.

        Returns:
            `RCMSchedulerOutput` or `tuple`:
                The denoised sample for the next step.
        """
        if self._step_index is None:
            self._init_step_index()

        assert self._step_index is not None
        
        # Get current and next sigma values (raw, unscaled) for rCM formula
        # Note: self.timesteps is scaled by 1000 for model input,
        # but we need raw values for the step formula
        t_cur = self.sigmas[self._step_index]
        # On the final step, t_next should be 0 (fully denoised)
        if self._step_index + 1 < len(self.sigmas):
            t_next = self.sigmas[self._step_index + 1]
        else:
            t_next = torch.tensor(0.0, device=sample.device, dtype=torch.float64)

        # Ensure we're working in float64 for precision
        sample = sample.to(torch.float64)
        model_output = model_output.to(torch.float64)

        # rCM update: x = (1 - t_next) * (x - t_cur * v_pred) + t_next * noise
        x_denoised = sample - t_cur * model_output
        
        # Generate noise for SDE sampling
        if isinstance(generator, list):
            generator = generator[0]
        noise = torch.randn(
            sample.shape,
            dtype=torch.float32,
            device="cpu",
            generator=generator,
        ).to(sample.device).to(torch.float64)

        prev_sample = (1 - t_next) * x_denoised + t_next * noise

        # Increment step counter
        self._step_index += 1

        # Cast back to model output dtype
        prev_sample = prev_sample.to(model_output.dtype)

        if not return_dict:
            return (prev_sample,)

        return RCMSchedulerOutput(prev_sample=prev_sample)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        """
        Add noise to samples (forward diffusion process).
        
        Not typically used for rCM inference, but provided for API compatibility.
        """
        raise NotImplementedError(
            "add_noise is not implemented for RCMScheduler. "
            "Use scale_noise for initializing the sampling process."
        )

    def __len__(self) -> int:
        return self.config.num_train_timesteps

# Entry point for model registry
EntryClass = RCMScheduler
