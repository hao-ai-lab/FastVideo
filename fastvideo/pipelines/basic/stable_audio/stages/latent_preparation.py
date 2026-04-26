# SPDX-License-Identifier: Apache-2.0
"""Stable Audio latent preparation — initial Gaussian noise.

Mirrors the upstream `generate_diffusion_cond` initial-noise step:

    torch.manual_seed(seed)
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

The k-diffusion sampler scales `noise` by `sigmas[0]` itself (so this
stage doesn't need to multiply by `init_noise_sigma`).
"""
from __future__ import annotations

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import VerificationResult


class StableAudioLatentPreparationStage(PipelineStage):

    def __init__(self, io_channels: int = 64, sample_size: int = 2097152) -> None:
        super().__init__()
        self.io_channels = io_channels
        # `sample_size` is the audio-domain length the official model was
        # trained for; latent length = sample_size // pretransform_downsampling
        # (= 2097152 / 2048 = 1024).
        self.sample_size = sample_size

    def verify_input(self, batch, fastvideo_args):
        return VerificationResult()

    def verify_output(self, batch, fastvideo_args):
        return VerificationResult()

    @torch.inference_mode()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        ext = batch.extra or {}
        # Reach into the conditioner to find the device.
        device = ext["cross_attn_cond"].device

        # Fixed downsampling ratio for Stable Audio Open 1.0 Oobleck VAE.
        downsampling_ratio = 2048
        latent_sample_size = self.sample_size // downsampling_ratio

        seed = batch.seed if batch.seed is not None else 0
        # Match upstream `generate_diffusion_cond`:
        #     torch.manual_seed(seed)
        #     noise = torch.randn([B, io_channels, sample_size], device=device)
        torch.manual_seed(int(seed))
        latents = torch.randn((1, self.io_channels, latent_sample_size), device=device)

        batch.latents = latents
        if batch.extra is None:
            batch.extra = {}
        batch.extra["seed"] = int(seed)
        batch.extra["downsampling_ratio"] = downsampling_ratio
        batch.extra["sample_size"] = self.sample_size
        return batch
