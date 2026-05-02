# SPDX-License-Identifier: Apache-2.0
"""Super-resolution latent preparation for daVinci-MagiHuman SR-540p."""
from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.utils import load_image
from diffusers.video_processor import VideoProcessor
from PIL import Image

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.basic.magi_human.stages.reference_image import (
    _resizecrop, )
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import VerificationResult


class ZeroSNRDDPMDiscretization:
    """Upstream ZeroSNR schedule used to corrupt interpolated SR latents."""

    def __init__(
        self,
        linear_start: float = 0.00085,
        linear_end: float = 0.0120,
        num_timesteps: int = 1000,
        shift_scale: float = 1.0,
        keep_start: bool = False,
        post_shift: bool = False,
    ) -> None:
        if keep_start and not post_shift:
            linear_start = linear_start / (shift_scale + (1 - shift_scale) * linear_start)
        self.num_timesteps = num_timesteps
        betas = torch.linspace(
            linear_start**0.5,
            linear_end**0.5,
            num_timesteps,
            dtype=torch.float64,
        )**2
        alphas = 1.0 - betas.numpy()
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.post_shift = post_shift
        self.shift_scale = shift_scale

        if not post_shift:
            self.alphas_cumprod = self.alphas_cumprod / (shift_scale + (1 - shift_scale) * self.alphas_cumprod)

    def __call__(
        self,
        n: int,
        do_append_zero: bool = True,
        device: str | torch.device = "cpu",
        flip: bool = False,
    ) -> torch.Tensor:
        sigmas = self.get_sigmas(n, device=device)
        if do_append_zero:
            sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        return torch.flip(sigmas, (0, )) if flip else sigmas

    def get_sigmas(
        self,
        n: int,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        if n < self.num_timesteps:
            timesteps = np.linspace(
                self.num_timesteps - 1,
                0,
                n,
                endpoint=False,
            ).astype(int)[::-1]
            alphas_cumprod = self.alphas_cumprod[timesteps]
        elif n == self.num_timesteps:
            alphas_cumprod = self.alphas_cumprod
        else:
            raise ValueError(f"n must be <= {self.num_timesteps}, got {n}")

        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        alphas_cumprod_sqrt = to_torch(alphas_cumprod).sqrt()
        alphas_cumprod_sqrt_0 = alphas_cumprod_sqrt[0].clone()
        alphas_cumprod_sqrt_T = alphas_cumprod_sqrt[-1].clone()

        alphas_cumprod_sqrt -= alphas_cumprod_sqrt_T
        alphas_cumprod_sqrt *= alphas_cumprod_sqrt_0 / (alphas_cumprod_sqrt_0 - alphas_cumprod_sqrt_T)

        if self.post_shift:
            alphas_cumprod_sqrt = (alphas_cumprod_sqrt**2 / (self.shift_scale +
                                                             (1 - self.shift_scale) * alphas_cumprod_sqrt**2))**0.5
        return torch.flip(alphas_cumprod_sqrt, (0, ))


class MagiHumanSRLatentPreparationStage(PipelineStage):
    """Upsample base latents, add SR noise, and refresh SR conditioning."""

    def __init__(
        self,
        vae: Any,
        vae_stride: tuple[int, int, int] = (4, 16, 16),
        patch_size: tuple[int, int, int] = (1, 2, 2),
        noise_value: int = 220,
        sr_audio_noise_scale: float = 0.7,
        sr_height: int = 512,
        sr_width: int = 896,
        vae_scale_factor: int = 16,
    ) -> None:
        super().__init__()
        self.vae = vae
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.noise_value = noise_value
        self.sr_audio_noise_scale = sr_audio_noise_scale
        self.sr_height = sr_height
        self.sr_width = sr_width
        self.sigmas = ZeroSNRDDPMDiscretization()(1000, do_append_zero=False, flip=True)
        self.video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor)

    def verify_input(self, batch, fastvideo_args):
        return VerificationResult()

    def verify_output(self, batch, fastvideo_args):
        return VerificationResult()

    @torch.inference_mode()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        device = batch.latents.device
        _, _, latent_t, _, _ = batch.latents.shape
        _, vh, vw = self.vae_stride
        _, pH, pW = self.patch_size
        latent_h = (self.sr_height // vh // pH) * pH
        latent_w = (self.sr_width // vw // pW) * pW
        actual_h = latent_h * vh
        actual_w = latent_w * vw

        latent_video = F.interpolate(
            batch.latents,
            size=(latent_t, latent_h, latent_w),
            mode="trilinear",
            align_corners=True,
        )
        if self.noise_value != 0:
            noise = torch.randn_like(latent_video, device=device)
            sigma = self.sigmas.to(device)[self.noise_value]
            latent_video = latent_video * sigma + noise * (1 - sigma**2)**0.5

        batch.latents = latent_video
        batch.audio_latents = (
            torch.randn_like(batch.audio_latents, device=batch.audio_latents.device) * self.sr_audio_noise_scale +
            batch.audio_latents * (1 - self.sr_audio_noise_scale))
        batch.height = actual_h
        batch.width = actual_w
        batch.magi_latent_T = latent_t
        batch.magi_latent_H = latent_h
        batch.magi_latent_W = latent_w

        if getattr(batch, "image_latent", None) is not None:
            batch.image_latent = self._encode_image(batch, actual_h, actual_w)
        return batch

    def _encode_image(
        self,
        batch: ForwardBatch,
        height: int,
        width: int,
    ) -> torch.Tensor:
        image = getattr(batch, "image", None) or batch.pil_image
        if image is None and batch.image_path is not None:
            image = load_image(batch.image_path)
        if image is None:
            raise ValueError("MagiHuman SR TI2V requires an image for SR re-encoding.")
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL image or image path, got {type(image)}")

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

        shift_factor = getattr(self.vae, "shift_factor", None)
        if shift_factor is not None:
            if isinstance(shift_factor, torch.Tensor):
                image_latent = image_latent - shift_factor.to(
                    image_latent.device,
                    image_latent.dtype,
                )
            else:
                image_latent = image_latent - shift_factor
        scaling_factor = getattr(self.vae, "scaling_factor", None)
        if scaling_factor is not None:
            if isinstance(scaling_factor, torch.Tensor):
                image_latent = image_latent * scaling_factor.to(
                    image_latent.device,
                    image_latent.dtype,
                )
            else:
                image_latent = image_latent * scaling_factor
        return image_latent.to(torch.float32)
