# SPDX-License-Identifier: Apache-2.0
"""Pack Diffusers-compatible VACE control latents (video/mask/reference)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.vision_utils import load_image, normalize, numpy_to_pt, pil_to_numpy, resize
from fastvideo.models.wan.vae import AutoencoderKLWan
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.utils import PRECISION_TO_TYPE


class WanVACEContextStage(PipelineStage):
    """Build ``control_hidden_states`` = concat(video_latents, mask_latents) for VACE."""

    def __init__(self, vae: AutoencoderKLWan) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        config = fastvideo_args.pipeline_config
        device = get_local_torch_device()
        vae_dtype = PRECISION_TO_TYPE[config.vae_precision]
        self.vae = self.vae.to(device)

        video = batch.video_latent.to(device=device, dtype=torch.float32)
        mask = batch.mask_video
        if mask is None:
            mask = torch.ones_like(video)
        else:
            mask = torch.clamp((mask.to(device=device, dtype=torch.float32) + 1) / 2, min=0, max=1)

        reference_images = batch.vace_reference_images or []
        conditioning_latents = self._prepare_video_latents(video, mask, reference_images, batch.generator, device,
                                                           vae_dtype)
        mask_latents = self._prepare_masks(mask, reference_images, config)
        batch.vace_control_latents = torch.cat([conditioning_latents, mask_latents], dim=1)
        batch.vace_num_reference_frames = len(reference_images)
        return batch

    def _encode_latent(self, pixels: torch.Tensor, vae_dtype: torch.dtype) -> torch.Tensor:
        latents_mean = torch.tensor(self.vae.latents_mean, device=pixels.device, dtype=torch.float32).view(
            1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(self.vae.latents_std, device=pixels.device, dtype=torch.float32).view(
            1, self.vae.config.z_dim, 1, 1, 1)
        encoded = self.vae.encode(pixels.to(dtype=vae_dtype)).mode()
        return ((encoded.float() - latents_mean) * latents_std).to(vae_dtype)

    def _prepare_video_latents(
        self,
        video: torch.Tensor,
        mask: torch.Tensor,
        reference_images: list[torch.Tensor],
        generator: torch.Generator | list[torch.Generator] | None,
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


def preprocess_vace_reference_images(
    references: list[str] | None,
    image_size: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    if not references:
        return []
    height, width = image_size
    processed = []
    for path in references:
        image = load_image(path)
        image = resize(image, height, width, resize_mode="default", resample="lanczos")
        tensor = numpy_to_pt(normalize(pil_to_numpy([image]))).squeeze(0)
        img_height, img_width = tensor.shape[-2:]
        scale = min(height / img_height, width / img_width)
        new_height, new_width = int(img_height * scale), int(img_width * scale)
        resized = F.interpolate(tensor.unsqueeze(0), size=(new_height, new_width), mode="bilinear",
                                align_corners=False).squeeze(0)
        top = (height - new_height) // 2
        left = (width - new_width) // 2
        canvas = torch.ones(3, height, width, device=device, dtype=dtype)
        canvas[:, top:top + new_height, left:left + new_width] = resized.to(device=device, dtype=dtype)
        processed.append(canvas)
    return processed
