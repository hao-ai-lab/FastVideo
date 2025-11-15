# SPDX-License-Identifier: Apache-2.0
"""
Timestep preparation stages for diffusion pipelines.

This module contains implementations of timestep preparation stages for diffusion pipelines.
"""

import inspect
import numpy as np

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
            scheduler.set_timesteps(timesteps=timesteps,
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


class LTXTimestepPreparationStage(TimestepPreparationStage):
    """
    Timestep preparation stage for LTX Video models
    Handles adaptive timestep shifting based on sequence length
    """

    def __init__(self, scheduler) -> None:
        super().__init__(scheduler)

    def calculate_shift(self,
                        image_seq_len: int,
                        base_seq_len: int = 256,
                        max_seq_len: int = 4096,
                        base_shift: float = 0.5,
                        max_shift: float = 1.15) -> float:
        """
        Calculate adaptive shift value based on sequence length.
        
        LTX uses dynamic shifting to adjust the timestep schedule based on
        the complexity of the generation task (sequence length).
        """
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Prepare timesteps for the LTX diffusion process.
        
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

        # Set default sigmas for LTX if not provided
        if sigmas is None:
            sigmas = np.linspace(1.0, 1 / num_inference_steps,
                                 num_inference_steps)
            batch.sigmas = sigmas

        # Prepare extra kwargs for set_timesteps
        extra_set_timesteps_kwargs = {}
        if n_tokens is not None and "n_tokens" in inspect.signature(
                scheduler.set_timesteps).parameters:
            extra_set_timesteps_kwargs["n_tokens"] = n_tokens

        # Handle adaptive shift for sequence-aware scheduling (LTX specific)
        if batch.sequence_length:
            mu = self.calculate_shift(
                batch.sequence_length,
                self.scheduler.config.base_image_seq_len,
                self.scheduler.config.max_image_seq_len,
                self.scheduler.config.base_shift,
                self.scheduler.config.max_shift,
            )
            if 'mu' in inspect.signature(
                    self.scheduler.set_timesteps).parameters:
                extra_set_timesteps_kwargs["mu"] = mu
                logger.info(
                    "Using adaptive shift mu=%.4f for sequence length %d", mu,
                    batch.sequence_length)

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
            scheduler.set_timesteps(timesteps=timesteps,
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
        """Verify LTX timestep preparation stage inputs."""
        result = VerificationResult()
        result.add_check("num_inference_steps", batch.num_inference_steps,
                         V.positive_int)
        result.add_check("timesteps", batch.timesteps, V.none_or_tensor)
        result.add_check("sigmas", batch.sigmas, V.none_or_list)
        result.add_check("n_tokens", batch.n_tokens, V.none_or_positive_int)
        # LTX specific: sequence_length is used for adaptive shift
        result.add_check("sequence_length", batch.sequence_length,
                         V.positive_int)
        return result

    def verify_output(self, batch: ForwardBatch,
                      fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify LTX timestep preparation stage outputs."""
        result = VerificationResult()
        result.add_check("timesteps", batch.timesteps,
                         [V.is_tensor, V.with_dims(1)])
        return result
