# SPDX-License-Identifier: Apache-2.0
"""
Timestep preparation stages for diffusion pipelines.

This module contains implementations of timestep preparation stages for diffusion pipelines.
"""

import inspect

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult

logger = init_logger(__name__)


class TimestepPreparationStage(PipelineStage):
    """
    Stage for preparing timesteps for the diffusion process.
    
    This stage handles the preparation of the timestep sequence that will be used
    during the diffusion process.
    """

    def __init__(self, scheduler) -> None:
        self.scheduler = scheduler

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Prepare timesteps for the diffusion process.
        
        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.
            
        Returns:
            The batch with prepared timesteps.
        """
        scheduler = self.scheduler
        device = get_local_torch_device()
        num_inference_steps = batch.num_inference_steps
        timesteps = batch.timesteps
        sigmas = batch.sigmas
        n_tokens = batch.n_tokens

        # Prepare extra kwargs for set_timesteps
        extra_set_timesteps_kwargs = {}
        if n_tokens is not None and "n_tokens" in inspect.signature(
                scheduler.set_timesteps).parameters:
            extra_set_timesteps_kwargs["n_tokens"] = n_tokens

        # Handle custom timesteps or sigmas
        if timesteps is not None and sigmas is not None:
            raise ValueError(
                "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
            )

        if timesteps is not None:
            accepts_timesteps = "timesteps" in inspect.signature(
                scheduler.set_timesteps).parameters
            if not accepts_timesteps:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" timestep schedules. Please check whether you are using the correct scheduler."
                )
            # Convert timesteps to CPU if it's a tensor (for numpy conversion in scheduler)
            if isinstance(timesteps, torch.Tensor):
                timesteps_for_scheduler = timesteps.cpu()
            else:
                timesteps_for_scheduler = timesteps
            scheduler.set_timesteps(timesteps=timesteps_for_scheduler,
                                    device=device,
                                    **extra_set_timesteps_kwargs)
            timesteps = scheduler.timesteps
        elif sigmas is not None:
            accept_sigmas = "sigmas" in inspect.signature(
                scheduler.set_timesteps).parameters
            if not accept_sigmas:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" sigmas schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(sigmas=sigmas,
                                    device=device,
                                    **extra_set_timesteps_kwargs)
            timesteps = scheduler.timesteps
        else:
            scheduler.set_timesteps(num_inference_steps,
                                    device=device,
                                    **extra_set_timesteps_kwargs)
            timesteps = scheduler.timesteps

        # Update batch with prepared timesteps
        batch.timesteps = timesteps

        return batch

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify timestep preparation stage inputs."""
        result = VerificationResult()
        result.add_check("num_inference_steps", batch.num_inference_steps,
                         V.positive_int)
        result.add_check("timesteps", batch.timesteps, V.none_or_tensor)
        result.add_check("sigmas", batch.sigmas, V.none_or_list)
        result.add_check("n_tokens", batch.n_tokens, V.none_or_positive_int)
        return result

    def verify_output(self, batch: ForwardBatch,
                      fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify timestep preparation stage outputs."""
        result = VerificationResult()
        result.add_check("timesteps", batch.timesteps,
                         [V.is_tensor, V.with_dims(1)])
        return result


class Cosmos25TimestepPreparationStage(TimestepPreparationStage):
    """Cosmos 2.5 timestep preparation with scheduler-specific kwargs."""

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        scheduler = self.scheduler
        device = get_local_torch_device()
        num_inference_steps = batch.num_inference_steps

        extra_kwargs: dict = {}
        sig = inspect.signature(scheduler.set_timesteps)
        if "shift" in sig.parameters:
            extra_kwargs["shift"] = fastvideo_args.pipeline_config.flow_shift
        # Prefer the canonical diffusers kwarg name if available.
        if "use_karras_sigmas" in sig.parameters:
            extra_kwargs["use_karras_sigmas"] = True
        elif "use_kerras_sigma" in sig.parameters:
            extra_kwargs["use_kerras_sigma"] = True

        scheduler.set_timesteps(num_inference_steps,
                                device=device,
                                **extra_kwargs)
        batch.timesteps = scheduler.timesteps
        return batch


class SD35TimestepPreparationStage(TimestepPreparationStage):
    """SD3/SD3.5 timestep preparation with optional dynamic shifting (mu).

    When the scheduler supports `use_dynamic_shifting`, this stage computes a
    resolution-dependent `mu` value and passes it to `set_timesteps()`.
    """

    @staticmethod
    def _calculate_mu(
        image_seq_len: int,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
    ) -> float:
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        return float(image_seq_len) * m + b

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        sig = inspect.signature(self.scheduler.set_timesteps)

        if "mu" in sig.parameters:
            cfg = getattr(self.scheduler, "config", None)
            use_dynamic = bool(
                getattr(cfg, "use_dynamic_shifting", False)
            ) if cfg is not None else False

            if use_dynamic:
                arch = fastvideo_args.pipeline_config.dit_config.arch_config
                vae_arch = fastvideo_args.pipeline_config.vae_config.arch_config
                patch_size = getattr(arch, "patch_size", None)
                spatial_ratio = getattr(vae_arch,
                                        "spatial_compression_ratio", None)

                if not isinstance(patch_size, int) or not isinstance(
                        spatial_ratio, int):
                    raise TypeError(
                        "SD3.5 dynamic shifting requires integer patch_size "
                        "and spatial_compression_ratio."
                    )
                if batch.height is None or batch.width is None:
                    raise ValueError(
                        "height/width must be set before timesteps.")

                h_lat = batch.height // spatial_ratio
                w_lat = batch.width // spatial_ratio
                image_seq_len = (h_lat // patch_size) * (w_lat // patch_size)

                base_seq_len = int(
                    getattr(cfg, "base_image_seq_len", 256))
                max_seq_len = int(
                    getattr(cfg, "max_image_seq_len", 4096))
                base_shift = float(getattr(cfg, "base_shift", 0.5))
                max_shift = float(getattr(cfg, "max_shift", 1.15))

                batch.n_tokens = None
                device = get_local_torch_device()
                self.scheduler.set_timesteps(
                    batch.num_inference_steps,
                    device=device,
                    mu=self._calculate_mu(
                        image_seq_len=image_seq_len,
                        base_seq_len=base_seq_len,
                        max_seq_len=max_seq_len,
                        base_shift=base_shift,
                        max_shift=max_shift,
                    ),
                )
                batch.timesteps = self.scheduler.timesteps
                return batch

        return super().forward(batch, fastvideo_args)
