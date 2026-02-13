# SPDX-License-Identifier: Apache-2.0
"""
Stable Audio latent preparation: sample initial noise for k-diffusion.
"""
from __future__ import annotations

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage


class StableAudioLatentPreparationStage(PipelineStage):
    """Prepare initial noise latents for Stable Audio denoising."""

    def __init__(self, pretransform) -> None:
        super().__init__()
        self.pretransform = pretransform

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        sample_rate = getattr(fastvideo_args.pipeline_config, "sample_rate",
                              44100) or 44100
        duration = batch.duration_seconds or 10.0
        sample_size = int(duration * sample_rate)
        latent_size = sample_size // self.pretransform.downsampling_ratio
        latent_channels = self.pretransform.encoded_channels

        batch_size = 1
        if batch.prompt is not None:
            batch_size = len(batch.prompt) if isinstance(batch.prompt,
                                                         list) else 1
        batch_size *= batch.num_videos_per_prompt

        device = get_local_torch_device()
        dtype = next(self.pretransform.parameters()).dtype

        seed = batch.seeds[0] if batch.seeds else (batch.seed or 0)
        generator = batch.generator
        if isinstance(generator, list):
            generator = generator[0] if generator else None
        if generator is None or str(generator.device) != str(device):
            generator = torch.Generator(device).manual_seed(seed)

        latents = torch.randn(
            (batch_size, latent_channels, latent_size),
            generator=generator,
            device=device,
            dtype=dtype,
        )

        batch.latents = latents
        batch.raw_latent_shape = latents.shape
        batch.extra["stable_audio_sample_size"] = sample_size
        return batch
