# SPDX-License-Identifier: Apache-2.0
"""
Latent preparation stage for diffusion pipelines.
"""
from diffusers.utils.torch_utils import randn_tensor

from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.models.vaes.common import ParallelTiledVAE
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines.stages.base import PipelineStage

logger = init_logger(__name__)


class LatentPreparationStage(PipelineStage):
    """
    Stage for preparing initial latent variables for the diffusion process.
    
    This stage handles the preparation of the initial latent variables that will be
    denoised during the diffusion process.
    """

    def __init__(self, scheduler, vae=None) -> None:
        super().__init__()
        self.scheduler = scheduler
        self.vae = vae

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Prepare initial latent variables for the diffusion process.
        
        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.
            
        Returns:
            The batch with prepared latent variables.
        """

        # Adjust video length based on VAE version if needed
        if hasattr(self, 'adjust_video_length'):
            batch = self.adjust_video_length(self.vae, batch, fastvideo_args)
        # Determine batch size
        if isinstance(batch.prompt, list):
            batch_size = len(batch.prompt)
        elif batch.prompt is not None:
            batch_size = 1
        else:
            batch_size = batch.prompt_embeds[0].shape[0]

        # Adjust batch size for number of videos per prompt
        batch_size *= batch.num_videos_per_prompt

        # Get required parameters
        dtype = batch.prompt_embeds[0].dtype
        device = batch.device
        generator = batch.generator
        latents = batch.latents
        num_frames = batch.num_frames
        height = batch.height
        width = batch.width

        # TODO(will): remove this once we add input/output validation for stages
        if height is None or width is None:
            raise ValueError("Height and width must be provided")

        assert fastvideo_args.num_channels_latents is not None
        assert fastvideo_args.vae_scale_factor is not None

        # Calculate latent shape
        shape = (
            batch_size,
            fastvideo_args.num_channels_latents,
            num_frames,
            height // fastvideo_args.vae_scale_factor,
            width // fastvideo_args.vae_scale_factor,
        )

        # Validate generator if it's a list
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # Generate or use provided latents
        if latents is None:
            latents = randn_tensor(shape,
                                   generator=generator,
                                   device=device,
                                   dtype=dtype)
        else:
            latents = latents.to(device)

        # Scale the initial noise if needed
        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma

        # Update batch with prepared latents
        batch.latents = latents

        return batch

    def adjust_video_length(self, vae: ParallelTiledVAE, batch: ForwardBatch,
                            fastvideo_args: FastVideoArgs) -> ForwardBatch:
        """
        Adjust video length based on VAE version.
        
        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.
            
        Returns:
            The batch with adjusted video length.
        """
        video_length = batch.num_frames
        temporal_scale_factor = vae.temporal_compression_ratio if vae is not None else 4
        # TODO
        batch.num_frames = (video_length - 1) // temporal_scale_factor + 1
        return batch
