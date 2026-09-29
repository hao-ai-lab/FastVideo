# SPDX-License-Identifier: Apache-2.0
"""Pack Diffusers-compatible VACE control latents (video/mask/reference)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.wan.vae import AutoencoderKLWan
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.utils import PRECISION_TO_TYPE


class WanVACEContextStage(PipelineStage):
    """Pack VACE control latents for the DiT ``control_hidden_states`` input.

    The output has ``vace_in_channels`` (96 by default):
    - channels 0-31: inactive + reactive video latents (16 channels each)
    - channels 32-95: downsampled mask latents aligned to the VAE latent grid

    Shape: ``[B, 96, T_latent, H_latent, W_latent]``. Values stay in VAE fp32 until
    the denoising stage casts once to the DiT dtype.
    """

    def __init__(self, vae: AutoencoderKLWan) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        config = fastvideo_args.pipeline_config
        device = get_local_torch_device()
        vae_dtype = PRECISION_TO_TYPE[config.vae_precision]
        original_device = next(self.vae.parameters()).device
        self.vae = self.vae.to(device)

        video = batch.video_latent.to(device=device, dtype=torch.float32)
        mask = batch.mask_video
        if mask is None:
            mask = torch.ones_like(video)
        else:
            mask = torch.clamp((mask.to(device=device, dtype=torch.float32) + 1) / 2, min=0, max=1)

        reference_images = batch.vace_reference_images or []
        conditioning_latents = self._prepare_video_latents(video, mask, reference_images, device, vae_dtype)
        mask_latents = self._prepare_masks(mask, reference_images, config)
        # Match Diffusers WanVACEPipeline: keep packed control in VAE fp32 until the
        # denoising stage casts once to DiT dtype immediately before transformer.
        batch.vace_control_latents = torch.cat([conditioning_latents, mask_latents], dim=1).to(device=device)
        # The pixels are fully encoded into the control latents; drop them so the
        # Wan denoising stage does not keep them or allocate ``video_padding``.
        batch.video_latent = None
        batch.mask_video = None
        if fastvideo_args.vae_cpu_offload:
            self.vae = self.vae.to(original_device)
        return batch

    def _normalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        arch = self.vae.config.arch_config
        shift = arch.shift_factor
        if shift is not None:
            latent = latent - shift.to(latent.device, latent.dtype)
        scale = arch.scaling_factor
        return latent * scale.to(latent.device, latent.dtype)

    def _encode_latent(self, pixels: torch.Tensor, vae_dtype: torch.dtype) -> torch.Tensor:
        encoded = self.vae.encode(pixels.to(dtype=vae_dtype)).mode()
        return self._normalize_latent(encoded.float()).to(vae_dtype)

    def _prepare_video_latents(
        self,
        video: torch.Tensor,
        mask: torch.Tensor,
        reference_images: list[torch.Tensor],
        device: torch.device,
        vae_dtype: torch.dtype,
    ) -> torch.Tensor:
        video = video.to(dtype=vae_dtype)
        mask = torch.where(mask > 0.5, 1.0, 0.0).to(dtype=vae_dtype)
        inactive = self._encode_latent(video * (1 - mask), vae_dtype)
        reactive = self._encode_latent(video * mask, vae_dtype)
        latents = torch.cat([inactive, reactive], dim=1)

        latent_list = []
        for latent in latents.unbind(0):
            for reference_image in reference_images:
                ref = reference_image.to(dtype=vae_dtype, device=device)[None, :, None, :, :]
                reference_latent = self._encode_latent(ref, vae_dtype).squeeze(0)
                reference_latent = torch.cat([reference_latent, torch.zeros_like(reference_latent)], dim=0)
                latent = torch.cat([reference_latent, latent], dim=1)
            latent_list.append(latent)
        return torch.stack(latent_list)

    def _prepare_masks(self, mask: torch.Tensor, reference_images: list[torch.Tensor],
                       config: PipelineConfig) -> torch.Tensor:
        patch_size = config.dit_config.arch_config.patch_size[1]
        temporal_ratio = config.vae_config.arch_config.temporal_compression_ratio
        spatial_ratio = config.vae_config.arch_config.scale_factor_spatial

        mask_list = []
        for mask_ in mask.unbind(0):
            num_channels, num_frames, height, width = mask_.shape
            new_num_frames = (num_frames + temporal_ratio - 1) // temporal_ratio
            new_height = height // (spatial_ratio * patch_size) * patch_size
            new_width = width // (spatial_ratio * patch_size) * patch_size
            mask_ = mask_[0]
            mask_ = mask_.view(num_frames, new_height, spatial_ratio, new_width, spatial_ratio)
            mask_ = mask_.permute(2, 4, 0, 1, 3).flatten(0, 1)
            mask_ = F.interpolate(mask_.unsqueeze(0),
                                  size=(new_num_frames, new_height, new_width),
                                  mode="nearest-exact").squeeze(0)
            if reference_images:
                padding = torch.zeros_like(mask_[:, :len(reference_images), :, :])
                mask_ = torch.cat([padding, mask_], dim=1)
            mask_list.append(mask_)
        return torch.stack(mask_list)

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("video_latent", batch.video_latent, V.not_none)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("vace_control_latents", batch.vace_control_latents, [V.is_tensor, V.with_dims(5)])
        return result
