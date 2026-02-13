# SPDX-License-Identifier: Apache-2.0
"""
GameCraft image-to-video encoding stage.

Encodes a reference image into gt_latents and conditioning_mask for
HunyuanGameCraft I2V generation. For T2V this stage is a no-op.
"""

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class GameCraftImageVAEEncodingStage(PipelineStage):
    """
    Stage for encoding a reference image into gt_latents and conditioning_mask
    for HunyuanGameCraft image-to-video generation.

    Official GameCraft I2V flow:
    1. VAE-encode the reference image -> [B, 16, 1, H_lat, W_lat]
    2. Scale by VAE scaling_factor (0.476986)
    3. Repeat to all temporal frames
    4. Zero out non-conditioned frames (first frame only for short videos,
       first half for longer autoregressive generation)
    5. Build a binary mask (1 = conditioned, 0 = generate)
    6. Store gt_latents and conditioning_mask on the batch for the denoising stage

    If no image is provided (T2V mode), this stage is a no-op; the denoising
    stage already falls back to zero gt_latents and zero mask.
    """

    def __init__(self, vae) -> None:
        super().__init__()
        self.vae = vae

    @torch.no_grad()
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """Encode reference image for I2V, or skip for T2V."""

        if batch.pil_image is None:
            # T2V mode: nothing to do; denoising stage handles the fallback
            return batch

        device = get_local_torch_device()
        image = batch.pil_image

        # ------------------------------------------------------------------
        # 1. Preprocess image to tensor [B, 3, 1, H, W]  (values in [-1, 1])
        # ------------------------------------------------------------------
        from torchvision import transforms

        target_height, target_width = batch.height, batch.width

        if isinstance(image, torch.Tensor):
            # Already a tensor (e.g. from InputValidationStage causal path)
            if image.dim() == 5:
                # [B, C, F, H, W] -> take first frame
                ref_pixel = image[:, :, :1]
            elif image.dim() == 4:
                ref_pixel = image.unsqueeze(
                    2)  # [B, C, H, W] -> [B, C, 1, H, W]
            else:
                raise ValueError(f"Unexpected image tensor dims: {image.dim()}")
            ref_pixel = ref_pixel.to(device=device, dtype=torch.float32)
        else:
            # PIL Image – resize, center-crop, normalize to [-1, 1]
            from PIL import Image as PILImage
            if not isinstance(image, PILImage.Image):
                import numpy as np
                if isinstance(image, np.ndarray):
                    image = PILImage.fromarray(image)

            original_w, original_h = image.size
            scale = max(target_width / original_w, target_height / original_h)
            resize_w = int(round(original_w * scale))
            resize_h = int(round(original_h * scale))

            ref_transform = transforms.Compose([
                transforms.Resize(
                    (resize_h, resize_w),
                    interpolation=transforms.InterpolationMode.LANCZOS,
                ),
                transforms.CenterCrop((target_height, target_width)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            ref_pixel = ref_transform(image)  # [3, H, W]
            ref_pixel = ref_pixel.unsqueeze(0).unsqueeze(2)  # [1, 3, 1, H, W]
            ref_pixel = ref_pixel.to(device=device, dtype=torch.float32)

        # ------------------------------------------------------------------
        # 2. VAE-encode
        # ------------------------------------------------------------------
        self.vae = self.vae.to(device)

        vae_dtype = PRECISION_TO_TYPE[
            fastvideo_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32) and not fastvideo_args.disable_autocast

        with torch.autocast(device_type="cuda",
                            dtype=vae_dtype,
                            enabled=vae_autocast_enabled):
            if fastvideo_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()
            if not vae_autocast_enabled:
                ref_pixel = ref_pixel.to(vae_dtype)
            encoder_output = self.vae.encode(ref_pixel)

        # Sample from the distribution (official uses .sample())
        ref_latents = encoder_output.latent_dist.sample().to(
            dtype=torch.float32)
        # Scale by VAE scaling factor (0.476986 for GameCraft)
        ref_latents.mul_(self.vae.config.scaling_factor)
        # ref_latents: [B, 16, 1, H_lat, W_lat]

        # ------------------------------------------------------------------
        # 3. Build gt_latents + conditioning_mask
        # ------------------------------------------------------------------
        # Get latent temporal dimension from batch latents (set by LatentPreparationStage)
        latent_frames = batch.latents.shape[
            2] if batch.latents is not None else ((batch.num_frames - 1) // 4 +
                                                  1)

        # Repeat to all frames
        gt_latents = ref_latents.repeat(1, 1, latent_frames, 1, 1)
        # [B, 16, T_lat, H_lat, W_lat]

        # Mask construction following official GameCraft logic:
        #   - Short videos (latent_frames <= 10): first frame conditioned
        #   - Longer videos: first half conditioned (autoregressive)
        mask = torch.ones(
            gt_latents.shape[0],
            1,
            gt_latents.shape[2],
            gt_latents.shape[3],
            gt_latents.shape[4],
            device=gt_latents.device,
            dtype=gt_latents.dtype,
        )

        if latent_frames <= 10:
            # I2V: only first frame is conditioned
            gt_latents[:, :, 1:, :, :] = 0.0
            mask[:, :, 1:, :, :] = 0.0
        else:
            # Autoregressive: first half conditioned
            half = latent_frames // 2
            gt_latents[:, :, half:, :, :] = 0.0
            mask[:, :, half:, :, :] = 0.0

        batch.gt_latents = gt_latents.to(device=device)
        batch.conditioning_mask = mask.to(device=device)

        # Offload
        if fastvideo_args.vae_cpu_offload:
            self.vae.to("cpu")

        return batch

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        # Stage is a no-op when pil_image is None, so nothing required
        return result

    def verify_output(self, batch: ForwardBatch,
                      fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        # gt_latents and conditioning_mask are only set for I2V
        return result
