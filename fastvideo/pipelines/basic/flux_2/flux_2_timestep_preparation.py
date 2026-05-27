# SPDX-License-Identifier: Apache-2.0
"""Flux2-specific timestep preparation."""

import inspect

import numpy as np
import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.timestep_preparation import TimestepPreparationStage


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """
    Resolution-dependent mu for Flux2 flow-match scheduler.
    From Black Forest Labs flux2 official repo: sampling.compute_empirical_mu.
    """
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


class Flux2TimestepPreparationStage(TimestepPreparationStage):
    """Flux2 timestep preparation matching the Diffusers Flux2 schedule."""

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        scheduler = self.scheduler
        device = get_local_torch_device()
        num_inference_steps = batch.num_inference_steps
        timesteps = batch.timesteps
        sigmas = batch.sigmas
        n_tokens = batch.n_tokens

        extra_set_timesteps_kwargs = {}
        if n_tokens is not None and "n_tokens" in inspect.signature(scheduler.set_timesteps).parameters:
            extra_set_timesteps_kwargs["n_tokens"] = n_tokens

        # Flux2/BFL: Diffusers' Flux2 pipeline passes a custom sigma grid and
        # always supplies the resolution-dependent mu when the scheduler accepts
        # it.
        scheduler_config = getattr(scheduler, "config", None)
        use_flow_sigmas = (getattr(scheduler_config, "use_flow_sigmas", False) if scheduler_config else False)
        if timesteps is None and sigmas is None and not use_flow_sigmas:
            sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)

        if "mu" in inspect.signature(scheduler.set_timesteps).parameters:
            if batch.n_tokens is not None:
                image_seq_len = batch.n_tokens
            else:
                h = (batch.height if isinstance(batch.height, int) else (batch.height[0] if batch.height else None))
                w = (batch.width if isinstance(batch.width, int) else (batch.width[0] if batch.width else None))
                vae_config = getattr(fastvideo_args.pipeline_config, "vae_config", None)
                if vae_config is not None:
                    arch = getattr(vae_config, "arch_config", None)
                    scale = (getattr(arch, "spatial_compression_ratio", 8) if arch else 8)
                else:
                    scale = 8
                image_seq_len = ((h // scale) * (w // scale) if h is not None and w is not None else 256)
            extra_set_timesteps_kwargs["mu"] = compute_empirical_mu(image_seq_len, num_inference_steps)

        if timesteps is not None and sigmas is not None:
            raise ValueError("Only one of `timesteps` or `sigmas` can be passed. "
                             "Please choose one to set custom values")

        if timesteps is not None:
            accepts_timesteps = "timesteps" in inspect.signature(scheduler.set_timesteps).parameters
            if not accepts_timesteps:
                raise ValueError(f"The current scheduler class {scheduler.__class__}'s "
                                 f"`set_timesteps` does not support custom timestep schedules.")
            timesteps_for_scheduler = (timesteps.cpu() if isinstance(timesteps, torch.Tensor) else timesteps)
            scheduler.set_timesteps(
                timesteps=timesteps_for_scheduler,
                device=device,
                **extra_set_timesteps_kwargs,
            )
            timesteps = scheduler.timesteps
        elif sigmas is not None:
            accept_sigmas = "sigmas" in inspect.signature(scheduler.set_timesteps).parameters
            if not accept_sigmas:
                raise ValueError(f"The current scheduler class {scheduler.__class__}'s "
                                 f"`set_timesteps` does not support custom sigmas schedules.")
            scheduler.set_timesteps(
                sigmas=sigmas,
                device=device,
                **extra_set_timesteps_kwargs,
            )
            timesteps = scheduler.timesteps
        else:
            scheduler.set_timesteps(
                num_inference_steps,
                device=device,
                **extra_set_timesteps_kwargs,
            )
            timesteps = scheduler.timesteps

        batch.timesteps = timesteps
        return batch
