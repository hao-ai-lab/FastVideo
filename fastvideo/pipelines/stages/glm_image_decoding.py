# SPDX-License-Identifier: Apache-2.0
"""
GLM-Image decoding stage.

Simple 2D image decoding using diffusers AutoencoderKL.
"""

import torch
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import VerificationResult


class GlmImageDecodingStage(PipelineStage):
    """Simple decoding stage for GLM-Image (2D image generation)."""

    def __init__(self, vae, pipeline=None):
        super().__init__()
        self.vae = vae
        self.pipeline = pipeline

    @torch.no_grad()
    def forward(self, batch: ForwardBatch,
                fastvideo_args: FastVideoArgs) -> ForwardBatch:
        latents = batch.latents
        device = latents.device

        # Squeeze temporal dimension if present [B, C, T, H, W] -> [B, C, H, W]
        if latents.dim() == 5:
            latents = latents.squeeze(2)

        # GLM-Image uses latent normalization like CogVideoX
        # SGLang implementation indicates we should use latents_std and latents_mean
        # and ignore the scaling_factor (0.18215) when these are present.
        if hasattr(self.vae.config,
                   'latents_mean') and self.vae.config.latents_mean is not None:
            latents_mean = torch.tensor(self.vae.config.latents_mean,
                                        device=device,
                                        dtype=latents.dtype).view(1, 16, 1, 1)
            latents_std = torch.tensor(self.vae.config.latents_std,
                                       device=device,
                                       dtype=latents.dtype).view(1, 16, 1, 1)
            # Un-normalize: latents * std + mean
            latents = latents * latents_std + latents_mean
        elif hasattr(self.vae.config, 'scaling_factor'):
            latents = latents / self.vae.config.scaling_factor

        # Ensure VAE is on the same device as latents
        self.vae.to(device)

        # Decode latents to image
        decoded = self.vae.decode(latents.to(self.vae.dtype))

        # Handle different return types
        # Handle different return types
        image = decoded.sample if hasattr(decoded, 'sample') else decoded

        # Normalize from [-1, 1] to [0, 1]
        image = (image / 2 + 0.5).clamp(0, 1)

        # Store as output (add fake temporal dim for compatibility)
        batch.output = image.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]

        return batch

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        return VerificationResult()  # Minimal verification

    def verify_output(self, batch: ForwardBatch,
                      fastvideo_args: FastVideoArgs) -> VerificationResult:
        return VerificationResult()  # Minimal verification
