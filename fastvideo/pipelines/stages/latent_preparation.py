# SPDX-License-Identifier: Apache-2.0
"""
Latent preparation stage for diffusion pipelines.
"""

from typing import Any

import torch
from diffusers.utils.torch_utils import randn_tensor

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.image_processor import ImageProcessor
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


class CosmosLatentPreparationStage(PipelineStage):
    """
    Cosmos-specific latent preparation stage that properly handles the tensor shapes
    and conditioning masks required by the Cosmos transformer.
    
    This stage replicates the logic from diffusers' Cosmos2VideoToWorldPipeline.prepare_latents()
    """

    def __init__(self, scheduler, transformer, vae=None) -> None:
        super().__init__()
        self.scheduler = scheduler
        self.transformer = transformer
        self.vae = vae

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
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
        # Force float32 for latent preparation
        dtype = torch.float32
        device = get_local_torch_device()
        generator = batch.generator
        latents = batch.latents
        num_frames = batch.num_frames
        height = batch.height
        width = batch.width

        if height is None or width is None:
            raise ValueError("Height and width must be provided")

        vae_scale_factor_spatial = 8
        vae_scale_factor_temporal = 4

        latent_height = height // 8
        latent_width = width // vae_scale_factor_spatial

        num_latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1

        # Cosmos transformer expects in_channels - 1 for the latent channels
        num_channels_latents = self.transformer.config.in_channels - 1

        shape = (batch_size, num_channels_latents, num_latent_frames,
                 latent_height, latent_width)

        init_latents = None
        conditioning_latents = None

        video = None

        if hasattr(batch, 'video') and batch.video is not None:
            video = batch.video
        elif hasattr(batch, 'pil_image') and batch.pil_image is not None:
            vae_scale_factor_spatial = 8
            image_processor = ImageProcessor(
                vae_scale_factor=vae_scale_factor_spatial)

            processed_image = image_processor.preprocess(
                batch.pil_image, height, width)

            # Add time dimension
            video = processed_image.unsqueeze(2)

            video = video.to(device=device, dtype=torch.bfloat16)
        elif hasattr(
                batch,
                'preprocessed_image') and batch.preprocessed_image is not None:
            # Convert preprocessed image to video format
            if isinstance(batch.preprocessed_image, torch.Tensor):
                if batch.preprocessed_image.dim(
                ) == 4:  # [B, C, H, W] -> [B, C, T, H, W]
                    video = batch.preprocessed_image.unsqueeze(2)
                elif batch.preprocessed_image.dim(
                ) == 5:  # Already [B, C, T, H, W]
                    video = batch.preprocessed_image
        else:
            logger.info(
                "CosmosLatentPreparationStage - No video input sources found")

        if video is not None:
            num_cond_frames = video.size(2)

            if num_cond_frames >= num_frames:
                # Take the last `num_frames` frames for conditioning
                num_cond_latent_frames = (num_frames -
                                          1) // vae_scale_factor_temporal + 1
                video = video[:, :, -num_frames:]
            else:
                num_cond_latent_frames = (num_cond_frames -
                                          1) // vae_scale_factor_temporal + 1
                num_padding_frames = num_frames - num_cond_frames
                last_frame = video[:, :, -1:]
                padding = last_frame.repeat(1, 1, num_padding_frames, 1, 1)
                video = torch.cat([video, padding], dim=2)

            if self.vae is not None:
                # Move VAE to correct device before encoding
                self.vae = self.vae.to(device)
                self.vae = self.vae.to(dtype=video.dtype)

                def retrieve_latents(
                        encoder_output: Any,
                        generator: Any | None = None) -> torch.Tensor:
                    if hasattr(encoder_output, "latent_dist"):
                        return encoder_output.latent_dist.sample(generator)
                    elif hasattr(encoder_output, "latents"):
                        return encoder_output.latents
                    elif hasattr(encoder_output, "sample"):
                        return encoder_output.sample(generator)
                    elif isinstance(encoder_output, torch.Tensor):
                        return encoder_output
                    else:
                        attrs = [
                            attr for attr in dir(encoder_output)
                            if not attr.startswith('_')
                        ]
                        raise AttributeError(
                            f"Could not access latents of provided encoder_output. Available attributes: {attrs}"
                        )

                if isinstance(generator, list):
                    init_latents = [
                        retrieve_latents(self.vae.encode(video[i].unsqueeze(0)),
                                         generator=torch.Generator(
                                             device="cpu").manual_seed(100))
                        for i in range(batch_size)
                    ]
                else:
                    init_latents = [
                        retrieve_latents(
                            self.vae.encode(vid.unsqueeze(0)),
                            torch.Generator(device="cpu").manual_seed(100))
                        for vid in video
                    ]

                init_latents = torch.cat(init_latents, dim=0).to(dtype)

                # Apply latent normalization
                if hasattr(self.vae.config, 'latents_mean') and hasattr(
                        self.vae.config, 'latents_std'):
                    latents_mean = torch.tensor(
                        self.vae.config.latents_mean).view(
                            1, self.vae.config.z_dim, 1, 1,
                            1).to(device, dtype)
                    latents_std = torch.tensor(
                        self.vae.config.latents_std).view(
                            1, self.vae.config.z_dim, 1, 1,
                            1).to(device, dtype)
                    init_latents = (init_latents - latents_mean
                                    ) / latents_std * self.scheduler.sigma_data

                conditioning_latents = init_latents

                # Offload VAE to CPU after encoding to save memory
                self.vae.to("cpu")
        else:
            num_cond_latent_frames = 0

        # Generate or use provided latents
        if latents is None:
            # Use float32 for randn_tensor
            latents = randn_tensor(
                shape,
                generator=torch.Generator(device="cpu").manual_seed(200),
                device=device,
                dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        latents = latents * self.scheduler.sigma_max

        padding_shape = (batch_size, 1, num_latent_frames, latent_height,
                         latent_width)
        ones_padding = latents.new_ones(padding_shape)
        zeros_padding = latents.new_zeros(padding_shape)

        cond_indicator = latents.new_zeros(1, 1, latents.size(2), 1, 1)
        cond_indicator[:, :, :num_cond_latent_frames] = 1.0
        cond_mask = cond_indicator * ones_padding + (
            1 - cond_indicator) * zeros_padding

        uncond_indicator = None
        uncond_mask = None
        if batch.do_classifier_free_guidance:
            uncond_indicator = latents.new_zeros(1, 1, latents.size(2), 1, 1)
            uncond_indicator[:, :, :num_cond_latent_frames] = 1.0
            uncond_mask = uncond_indicator * ones_padding + (
                1 - uncond_indicator) * zeros_padding

        batch.latents = latents
        batch.raw_latent_shape = latents.shape

        batch.conditioning_latents = conditioning_latents
        batch.cond_indicator = cond_indicator
        batch.uncond_indicator = uncond_indicator
        batch.cond_mask = cond_mask
        batch.uncond_mask = uncond_mask

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

    def verify_output(self, batch: ForwardBatch,
                      fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify latent preparation stage outputs."""
        result = VerificationResult()
        result.add_check("latents", batch.latents,
                         [V.is_tensor, V.with_dims(5)])
        result.add_check("raw_latent_shape", batch.raw_latent_shape, V.is_tuple)
        return result
