# SPDX-License-Identifier: Apache-2.0
"""
LongCat Image VAE Encoding Stage for I2V generation.

This stage handles encoding a single input image to latent space with
LongCat-specific normalization for I2V conditioning.
"""

import PIL
import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.vision_utils import (normalize, numpy_to_pt, pil_to_numpy,
                                           resize)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class LongCatImageVAEEncodingStage(PipelineStage):
    """
    Encode input image to latent space for I2V conditioning.
    
    This stage:
    1. Preprocesses image to match target dimensions
    2. Encodes via VAE to latent space
    3. Applies LongCat-specific normalization
    4. Stores latent and calculates num_cond_latents
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """Encode image to latent for I2V conditioning."""

        # 1. Get image from batch
        image = batch.pil_image  # PIL.Image
        if image is None:
            raise ValueError("pil_image must be provided for I2V")

        if not isinstance(image, PIL.Image.Image):
            raise TypeError(f"pil_image must be PIL.Image, got {type(image)}")

        # 2. Get target dimensions
        height = batch.height
        width = batch.width

        if height is None or width is None:
            raise ValueError("height and width must be set for I2V")

        # 3. Preprocess image
        image = resize(image, height, width, resize_mode="default")
        image = pil_to_numpy(image)
        image = numpy_to_pt(image)
        image = normalize(image)  # to [-1, 1]

        # 4. Add temporal dimension
        # After numpy_to_pt: [1, C, H, W] (batch already added by pil_to_numpy)
        # Add T dimension: [1, C, H, W] -> [1, C, 1, H, W] = [B, C, T, H, W]
        image = image.unsqueeze(2)
        image = image.to(get_local_torch_device(), dtype=torch.float32)

        # 5. Encode via VAE
        self.vae = self.vae.to(get_local_torch_device())

        # Setup VAE precision
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
                image = image.to(vae_dtype)

            with torch.no_grad():
                encoder_output = self.vae.encode(image)
                latent = self.retrieve_latents(encoder_output, batch.generator)

        # 6. Apply LongCat-specific normalization
        # Formula: (latents - mean) / std
        latent = self.normalize_latents(latent)

        # 7. Calculate num_cond_latents
        # Formula: 1 + (num_cond_frames - 1) // vae_temporal_scale
        # For single image (num_cond_frames=1): always 1 latent frame
        num_cond_frames = 1  # Single image
        vae_temporal_scale = self.vae.config.scale_factor_temporal
        batch.num_cond_latents = 1 + (num_cond_frames - 1) // vae_temporal_scale

        # 8. Store in batch
        batch.image_latent = latent
        batch.num_cond_frames = 1

        logger.info(f"I2V: Encoded image to latent shape {latent.shape}, "
                    f"num_cond_latents={batch.num_cond_latents}")

        # Offload VAE if needed
        if fastvideo_args.vae_cpu_offload:
            self.vae.to("cpu")

        return batch

    def retrieve_latents(self, encoder_output, generator):
        """Sample from VAE posterior."""
        # WAN VAE returns an object with .sample() method
        if hasattr(encoder_output, 'sample'):
            return encoder_output.sample(generator)
        elif hasattr(encoder_output, 'latent_dist'):
            return encoder_output.latent_dist.sample(generator)
        elif hasattr(encoder_output, 'latents'):
            return encoder_output.latents
        else:
            raise AttributeError("Could not access latents from encoder output")

    def normalize_latents(self, latents):
        """
        Apply LongCat-specific latent normalization.
        
        Formula: (latents - mean) / std
        
        This matches the original LongCat implementation and is DIFFERENT
        from standard VAE scaling (which uses scaling_factor).
        """
        if not hasattr(self.vae.config, 'latents_mean') or not hasattr(
                self.vae.config, 'latents_std'):
            raise ValueError(
                "VAE config must have 'latents_mean' and 'latents_std' "
                "for LongCat normalization")

        latents_mean = torch.tensor(self.vae.config.latents_mean).view(
            1, self.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)

        latents_std = torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)

        return (latents - latents_mean) / latents_std
