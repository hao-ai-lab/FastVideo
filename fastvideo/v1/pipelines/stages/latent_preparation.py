# SPDX-License-Identifier: Apache-2.0
"""
Latent preparation stage for diffusion pipelines.
"""

from typing import Dict

from diffusers.utils.torch_utils import randn_tensor

from fastvideo.v1.distributed import get_torch_device
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines.stages.base import PipelineStage
from fastvideo.v1.pipelines.stages.validators import StageValidators as V

logger = init_logger(__name__)


class LatentPreparationStage(PipelineStage):
    """
    Stage for preparing initial latent variables for the diffusion process.
    
    This stage handles the preparation of the initial latent variables that will be
    denoised during the diffusion process.
    """

    def __init__(self, scheduler, transformer) -> None:
        super().__init__()
        self.scheduler = scheduler
        self.transformer = transformer

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

        latent_num_frames = None
        # Adjust video length based on VAE version if needed
        if hasattr(self, 'adjust_video_length'):
            latent_num_frames = self.adjust_video_length(batch, fastvideo_args)
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
        device = get_torch_device()
        generator = batch.generator
        latents = batch.latents
        num_frames = latent_num_frames if latent_num_frames is not None else batch.num_frames
        height = batch.height
        width = batch.width

        # TODO(will): remove this once we add input/output validation for stages
        if height is None or width is None:
            raise ValueError("Height and width must be provided")

        # Calculate latent shape
        shape = (
            batch_size,
            self.transformer.num_channels_latents,
            num_frames,
            height // fastvideo_args.pipeline_config.vae_config.arch_config.
            spatial_compression_ratio,
            width // fastvideo_args.pipeline_config.vae_config.arch_config.
            spatial_compression_ratio,
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
        batch.raw_latent_shape = latents.shape

        return batch

    def adjust_video_length(self, batch: ForwardBatch,
                            fastvideo_args: FastVideoArgs) -> int:
        """
        Adjust video length based on VAE version.
        
        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.
            
        Returns:
            The batch with adjusted video length.
        """
        video_length = batch.num_frames
        use_temporal_scaling_frames = fastvideo_args.pipeline_config.vae_config.use_temporal_scaling_frames
        if use_temporal_scaling_frames:
            temporal_scale_factor = fastvideo_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
            latent_num_frames = (video_length - 1) // temporal_scale_factor + 1
        else:  # stepvideo only
            latent_num_frames = video_length // 17 * 3
        return int(latent_num_frames)

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> Dict[str, bool]:
        """Verify latent preparation stage inputs."""
        return {
            # Prompt or embeddings for determining batch size
            "prompt_or_embeds": (V.string_or_list_strings(batch.prompt)
                                 or V.list_not_empty(batch.prompt_embeds)),
            # Text embeddings with proper dtype
            "prompt_embeds":
            (V.list_not_empty(batch.prompt_embeds)
             and all(V.is_tensor(emb) for emb in batch.prompt_embeds)),
            # Number of videos per prompt
            "num_videos_per_prompt":
            V.positive_int(batch.num_videos_per_prompt),
            # Random generators
            "generator":
            V.generator_or_list_generators(batch.generator),
            # Video dimensions
            "num_frames":
            V.positive_int(batch.num_frames),
            "height":
            V.positive_int(batch.height) and V.divisible_by(batch.height, 8),
            "width":
            V.positive_int(batch.width) and V.divisible_by(batch.width, 8),
            # Optional initial latents
            "latents": (batch.latents is None or V.is_tensor(batch.latents)),
        }

    def verify_output(self, batch: ForwardBatch,
                      fastvideo_args: FastVideoArgs) -> Dict[str, bool]:
        """Verify latent preparation stage outputs."""
        return {
            # Prepared latents: [batch_size, channels, frames, height_latents, width_latents]
            "latents":
            V.is_tensor(batch.latents) and V.tensor_with_dims(batch.latents, 5),
            # Raw latent shape tuple
            "raw_latent_shape":
            isinstance(batch.raw_latent_shape, tuple),
        }
