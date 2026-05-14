# SPDX-License-Identifier: Apache-2.0
"""Reference-image encoding for MagiHuman TI2V.

The upstream daVinci-MagiHuman TI2V path encodes the user image through the
Wan VAE and overwrites the first denoising latent frame with that clean latent
at every step. This stage mirrors `MagiEvaluator.encode_image` and stashes the
normalized latent on `batch.image_latent` for the latent-prep and denoise stages.
"""
from __future__ import annotations

from typing import Any

import torch
from diffusers.utils import load_image
from diffusers.video_processor import VideoProcessor
from PIL import Image

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import VerificationResult


def _resizecrop(image: Image.Image, height: int, width: int) -> Image.Image:
    """Mirror upstream `resizecrop`: center-crop to target aspect ratio."""
    current_width, current_height = image.size
    if current_width == width and current_height == height:
        return image
    if current_height / current_width > height / width:
        new_width = int(current_width)
        new_height = int(new_width * height / width)
    else:
        new_height = int(current_height)
        new_width = int(new_height * width / height)
    left = (current_width - new_width) / 2
    top = (current_height - new_height) / 2
    right = (current_width + new_width) / 2
    bottom = (current_height + new_height) / 2
    return image.crop((left, top, right, bottom))


class MagiHumanReferenceImageStage(PipelineStage):
    """Encode a TI2V reference image into the first-frame video latent."""

    def __init__(self, vae: Any, vae_scale_factor: int = 16) -> None:
        super().__init__()
        self.vae = vae
        self.video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor)

    def verify_input(self, batch, fastvideo_args):
        return VerificationResult()

    def verify_output(self, batch, fastvideo_args):
        return VerificationResult()

    @torch.inference_mode()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        image = getattr(batch, "image", None) or batch.pil_image
        if image is None and batch.image_path is not None:
            image = load_image(batch.image_path)
        if image is None:
            raise ValueError("MagiHuman TI2V requires `image_path` or `pil_image`.")
        if not isinstance(image, Image.Image):
            raise TypeError(f"MagiHuman TI2V expects a PIL image or image path, got {type(image)}")
        if batch.height is None or batch.width is None:
            raise ValueError("MagiHuman TI2V requires concrete height and width before image encoding.")

        height = int(batch.height)
        width = int(batch.width)
        device = get_local_torch_device()

        image = _resizecrop(image.convert("RGB"), height, width)
        image_tensor = self.video_processor.preprocess(
            image,
            height=height,
            width=width,
        ).to(device=device, dtype=torch.float32)
        image_tensor = image_tensor.unsqueeze(2)

        self.vae = self.vae.to(device)
        encoded = self.vae.encode(image_tensor)
        image_latent = encoded.mean if hasattr(encoded, "mean") else encoded

        # FastVideo's Wan VAE returns unnormalized posterior means; upstream
        # `WanVAE.encode` applies `(mu - mean) / std` before returning.
        shift_factor = getattr(self.vae, "shift_factor", None)
        if shift_factor is not None:
            if isinstance(shift_factor, torch.Tensor):
                image_latent = image_latent - shift_factor.to(image_latent.device, image_latent.dtype)
            else:
                image_latent = image_latent - shift_factor
        scaling_factor = getattr(self.vae, "scaling_factor", None)
        if scaling_factor is not None:
            if isinstance(scaling_factor, torch.Tensor):
                image_latent = image_latent * scaling_factor.to(image_latent.device, image_latent.dtype)
            else:
                image_latent = image_latent * scaling_factor

        batch.image_latent = image_latent.to(torch.float32)
        return batch
