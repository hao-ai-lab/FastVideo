# SPDX-License-Identifier: Apache-2.0
"""
Latent preparation stage for diffusion pipelines.
"""

from diffusers.utils.torch_utils import randn_tensor

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult

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
        device = get_local_torch_device()
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
        sum_value = latents.float().sum().item()
        logger.info(f"LatentPreparationStage: latents sum = {sum_value:.6f}")
        # Write to output file
        with open("/workspace/FastVideo/fastvideo_hidden_states.log", "a") as f:
            f.write(f"LatentPreparationStage: latents sum = {sum_value:.6f}\n")

        return batch


class CosmosLatentPreparationStage(PipelineStage):
    """
    Cosmos-specific latent preparation stage that properly handles the tensor shapes
    and conditioning masks required by the Cosmos transformer.
    
    This stage replicates the logic from diffusers' Cosmos2VideoToWorldPipeline.prepare_latents()
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
        Prepare latents for Cosmos model with proper shapes and conditioning masks.
        
        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.
            
        Returns:
            The batch with prepared latent variables and conditioning masks.
        """
        
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
        device = get_local_torch_device()
        generator = batch.generator
        latents = batch.latents
        num_frames = batch.num_frames
        height = batch.height
        width = batch.width

        if height is None or width is None:
            raise ValueError("Height and width must be provided")

        # Calculate Cosmos-specific dimensions
        # Use the same VAE scale factors as diffusers to match their latent shapes
        # Based on diffusers pipeline: lines 205-206
        vae_scale_factor_spatial = 8  # Standard spatial compression (matches diffusers)
        vae_scale_factor_temporal = 4  # Temporal compression (matches diffusers default)
        
        # Use same formula as diffusers cosmos pipeline
        num_latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1
        
        logger.info(f"CosmosLatentPreparationStage - input frames: {num_frames}, latent frames: {num_latent_frames}")
        latent_height = height // vae_scale_factor_spatial
        latent_width = width // vae_scale_factor_spatial
        
        # Cosmos transformer expects in_channels - 1 for the latent channels
        num_channels_latents = self.transformer.config.in_channels - 1
        
        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)

        logger.info(f"CosmosLatentPreparationStage - preparing latents with shape: {shape}")

        # Generate or use provided latents
        if latents is None:
            latents = randn_tensor(shape,
                                 generator=generator,
                                 device=device,
                                 dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        # Scale latents by sigma_max (Cosmos-specific)
        latents = latents * self.scheduler.sigma_max

        # Create conditioning masks (for video-to-world generation)
        padding_shape = (batch_size, 1, num_latent_frames, latent_height, latent_width)
        ones_padding = latents.new_ones(padding_shape)
        zeros_padding = latents.new_zeros(padding_shape)

        # For now, create empty conditioning (no conditioning frames)
        # TODO: Implement proper conditioning for video-to-world
        num_cond_latent_frames = 0
        cond_indicator = latents.new_zeros(1, 1, latents.size(2), 1, 1)
        if num_cond_latent_frames > 0:
            cond_indicator[:, :, :num_cond_latent_frames] = 1.0
        cond_mask = cond_indicator * ones_padding + (1 - cond_indicator) * zeros_padding

        # For classifier-free guidance
        uncond_indicator = None
        uncond_mask = None
        if batch.do_classifier_free_guidance:
            uncond_indicator = latents.new_zeros(1, 1, latents.size(2), 1, 1)
            if num_cond_latent_frames > 0:
                uncond_indicator[:, :, :num_cond_latent_frames] = 1.0
            uncond_mask = uncond_indicator * ones_padding + (1 - uncond_indicator) * zeros_padding

        # Store in batch
        batch.latents = latents
        batch.raw_latent_shape = latents.shape
        sum_value = latents.float().sum().item()
        logger.info(f"CosmosLatentPreparationStage: latents sum = {sum_value:.6f}, shape = {latents.shape}, sigma_max = {self.scheduler.sigma_max}")
        # Write to output file
        with open("/workspace/FastVideo/fastvideo_hidden_states.log", "a") as f:
            f.write(f"CosmosLatentPreparationStage: latents sum = {sum_value:.6f}, shape = {latents.shape}, sigma_max = {self.scheduler.sigma_max}\n")
        
        # Store Cosmos-specific conditioning data
        batch.conditioning_latents = None  # No conditioning frames for now
        batch.cond_indicator = cond_indicator
        batch.uncond_indicator = uncond_indicator
        batch.cond_mask = cond_mask
        batch.uncond_mask = uncond_mask

        logger.info(f"CosmosLatentPreparationStage - final latents shape: {latents.shape}")
        logger.info(f"CosmosLatentPreparationStage - cond_mask shape: {cond_mask.shape}")

        return batch

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify Cosmos latent preparation stage inputs."""
        result = VerificationResult()
        result.add_check(
            "prompt_or_embeds", None, lambda _: V.string_or_list_strings(
                batch.prompt) or V.list_not_empty(batch.prompt_embeds))
        result.add_check("prompt_embeds", batch.prompt_embeds,
                         V.list_of_tensors)
        result.add_check("num_videos_per_prompt", batch.num_videos_per_prompt,
                         V.positive_int)
        result.add_check("generator", batch.generator,
                         V.generator_or_list_generators)
        result.add_check("num_frames", batch.num_frames, V.positive_int)
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        result.add_check("latents", batch.latents, V.none_or_tensor)
        return result

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
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify latent preparation stage inputs."""
        result = VerificationResult()
        result.add_check(
            "prompt_or_embeds", None, lambda _: V.string_or_list_strings(
                batch.prompt) or V.list_not_empty(batch.prompt_embeds))
        result.add_check("prompt_embeds", batch.prompt_embeds,
                         V.list_of_tensors)
        result.add_check("num_videos_per_prompt", batch.num_videos_per_prompt,
                         V.positive_int)
        result.add_check("generator", batch.generator,
                         V.generator_or_list_generators)
        result.add_check("num_frames", batch.num_frames, V.positive_int)
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        result.add_check("latents", batch.latents, V.none_or_tensor)
        return result

    def verify_output(self, batch: ForwardBatch,
                      fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify latent preparation stage outputs."""
        result = VerificationResult()
        result.add_check("latents", batch.latents,
                         [V.is_tensor, V.with_dims(5)])
        result.add_check("raw_latent_shape", batch.raw_latent_shape, V.is_tuple)
        return result
