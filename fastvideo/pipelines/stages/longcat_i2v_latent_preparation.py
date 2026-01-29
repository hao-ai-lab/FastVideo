# SPDX-License-Identifier: Apache-2.0
"""
LongCat I2V Latent Preparation Stage.

This stage prepares latents with image conditioning for the first frame.
"""

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.latent_preparation import LatentPreparationStage

logger = init_logger(__name__)


class LongCatI2VLatentPreparationStage(LatentPreparationStage):
    """
    Prepare latents with image conditioning for first frame.
    
    This stage:
    1. Generates random noise for all frames
    2. Replaces first latent frame with encoded image latent
    3. Marks conditioning information in batch
    """

    # Uses parent __init__ - no need for additional constructor

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """Prepare latents with I2V conditioning."""

        # IMPORTANT: Skip if latents already prepared (e.g., by refinement init stage)
        # The refine_init stage encodes stage1 video and mixes with noise - don't overwrite!
        if batch.latents is not None:
            logger.info(
                "I2V Latent Prep: Skipping - latents already prepared "
                "(shape=%s), likely from refinement stage", batch.latents.shape)
            return batch

        # 1. Calculate dimensions
        num_frames = batch.num_frames
        height = batch.height
        width = batch.width

        # Get VAE compression factors
        # IMPORTANT: Use VAE's temporal compression (4), NOT transformer's patch_size[0] (1)
        vae_temporal_scale = fastvideo_args.pipeline_config.vae_config.arch_config.scale_factor_temporal
        vae_spatial_scale = fastvideo_args.pipeline_config.vae_config.arch_config.scale_factor_spatial

        num_latent_frames = (num_frames - 1) // vae_temporal_scale + 1
        latent_height = height // vae_spatial_scale
        latent_width = width // vae_spatial_scale

        num_channels = self.transformer.config.in_channels

        logger.info(
            "I2V Latent Prep: num_frames=%s, num_latent_frames=%s "
            "(vae_temporal_scale=%s), latent_shape=(%s, %s)", num_frames,
            num_latent_frames, vae_temporal_scale, latent_height, latent_width)

        # 2. Generate random noise for all frames
        # batch_size might not be set, default to 1
        batch_size = batch.batch_size if batch.batch_size is not None else 1
        shape = (batch_size, num_channels, num_latent_frames, latent_height,
                 latent_width)

        # Handle generator - may be a list for batch handling
        generator = batch.generator
        if isinstance(generator, list):
            generator = generator[0] if generator else None

        # torch.randn requires specific argument order: size, generator, dtype
        latents = torch.randn(*shape,
                              generator=generator).to(get_local_torch_device(),
                                                      dtype=torch.float32)

        # 3. Replace first frame with conditioned image latent
        if batch.image_latent is not None:
            num_cond_latents = batch.num_cond_latents
            latents[:, :, :
                    num_cond_latents] = batch.image_latent[:, :, :
                                                           num_cond_latents]

            logger.info(
                "I2V: Replaced first %s latent frame(s) with image conditioning",
                num_cond_latents)
        else:
            logger.warning(
                "No image_latent found in batch, proceeding without conditioning"
            )

        # 4. Store in batch
        batch.latents = latents

        # Required by base class output validator
        batch.raw_latent_shape = (num_latent_frames, latent_height,
                                  latent_width)

        return batch
