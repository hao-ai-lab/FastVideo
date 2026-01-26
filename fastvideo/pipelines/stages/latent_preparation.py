# SPDX-License-Identifier: Apache-2.0
"""
Latent preparation stage for diffusion pipelines.
"""

from typing import Any

import numpy as np
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

    def __init__(self,
                 scheduler,
                 transformer,
                 use_btchw_layout: bool = False) -> None:
        super().__init__()
        self.scheduler = scheduler
        self.transformer = transformer
        self.use_btchw_layout = use_btchw_layout

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
        # Determine batch size; fall back to action/image inputs when no text encoder is present
        if not batch.prompt_embeds:
            if batch.keyboard_cond is not None:
                batch_size = batch.keyboard_cond.shape[0]
            elif batch.mouse_cond is not None:
                batch_size = batch.mouse_cond.shape[0]
            elif batch.image_embeds:
                batch_size = batch.image_embeds[0].shape[0]
            else:
                batch_size = 1
        elif isinstance(batch.prompt, list):
            batch_size = len(batch.prompt)
        elif batch.prompt is not None:
            batch_size = 1
        else:
            batch_size = batch.prompt_embeds[0].shape[0]

        # Adjust batch size for number of videos per prompt
        batch_size *= batch.num_videos_per_prompt

        # Get required parameters
        if not batch.prompt_embeds:
            # Create a dummy zero-length text embedding to satisfy downstream checks.
            # Matrix-Game models have text_dim=0 and ignore encoder_hidden_states.
            transformer_dtype = next(self.transformer.parameters()).dtype
            device = get_local_torch_device()
            dummy_prompt = torch.zeros(batch_size,
                                       0,
                                       self.transformer.hidden_size,
                                       device=device,
                                       dtype=transformer_dtype)
            batch.prompt_embeds = [dummy_prompt]
            batch.negative_prompt_embeds = []
            batch.do_classifier_free_guidance = False
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
        bcthw_shape: tuple[int, ...] | None = None
        if self.use_btchw_layout:
            shape = (
                batch_size,
                num_frames,
                self.transformer.num_channels_latents,
                height // fastvideo_args.pipeline_config.vae_config.arch_config.
                spatial_compression_ratio,
                width // fastvideo_args.pipeline_config.vae_config.arch_config.
                spatial_compression_ratio,
            )
            bcthw_shape = tuple(shape[i] for i in [0, 2, 1, 3, 4])
        else:
            shape = (
                batch_size,
                self.transformer.num_channels_latents,
                num_frames,
                height // fastvideo_args.pipeline_config.vae_config.arch_config.
                spatial_compression_ratio,
                width // fastvideo_args.pipeline_config.vae_config.arch_config.
                spatial_compression_ratio,
            )
            bcthw_shape = shape

        # Validate generator if it's a list
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        # Generate or use provided latents
        if latents is None:
            # Handle batch processing: if generator is a list, generate latents for each item separately
            # This ensures each batch item gets unique randomness from its own generator
            if isinstance(generator, list) and len(generator) > 1:
                # Generate latents for each batch item with its own generator
                latents_list = []
                for b in range(batch_size):
                    # Generate latents for single item: [1, C, T, H, W] or [1, T, C, H, W] depending on layout
                    single_shape = (1,) + shape[1:]
                    single_latents = randn_tensor(
                        single_shape,
                        generator=generator[b],
                        device=device,
                        dtype=dtype,
                    )
                    latents_list.append(single_latents)
                # Concatenate along batch dimension
                latents = torch.cat(latents_list, dim=0)
            else:
                # Single generator or None - use standard path
                gen = generator[0] if isinstance(generator, list) and len(generator) > 0 else generator
                latents = randn_tensor(
                    shape,
                    generator=gen,
                    device=device,
                    dtype=dtype,
                )
            if hasattr(self.scheduler, "init_noise_sigma"):
                latents = latents * self.scheduler.init_noise_sigma
        else:
            # Pre-initialized latents:
            # - For LongCat refine (refine_from or stage1_video present), we should not re-scale by init_noise_sigma.
            # - For other models, keep the original behavior.
            latents = latents.to(device)
            is_longcat_refine = (batch.refine_from
                                 is not None) or (batch.stage1_video
                                                  is not None)
            if (not is_longcat_refine) and hasattr(self.scheduler,
                                                   "init_noise_sigma"):
                latents = latents * self.scheduler.init_noise_sigma

        # Update batch with prepared latents
        batch.latents = latents
        batch.raw_latent_shape = bcthw_shape

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
            "prompt_or_embeds", None,
            lambda _: V.string_or_list_strings(batch.prompt) or not batch.
            prompt_embeds or V.list_not_empty(batch.prompt_embeds))
        if batch.prompt_embeds:
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


class Cosmos25LatentPreparationStage(CosmosLatentPreparationStage):
    """Latent preparation for Cosmos 2.5 DiT input conventions."""

    @staticmethod
    def _arch_invariant_randn(
        shape: tuple[int, ...],
        *,
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Architecture-invariant RNG (matches cosmos_predict2.misc.arch_invariant_rand)."""
        rng = np.random.RandomState(seed)
        arr = rng.standard_normal(shape).astype(np.float32)
        return torch.from_numpy(arr).to(device=device, dtype=dtype)

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        # Differences vs `CosmosLatentPreparationStage`: channel convention, seed usage,
        # and `padding_mask` for concat_padding_mask=True.

        # Determine batch size
        if isinstance(batch.prompt, list):
            batch_size = len(batch.prompt)
        elif batch.prompt is not None:
            batch_size = 1
        else:
            batch_size = batch.prompt_embeds[0].shape[0]

        batch_size *= batch.num_videos_per_prompt

        # Match `compare_pipelines.py`: initialize noise in fp32, then run the
        # denoising computation in bf16.
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

        # Cosmos 2.5 convention: transformer in_channels == latent channels
        num_channels_latents = self.transformer.config.in_channels

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
            video = processed_image.unsqueeze(2)
            video = video.to(device=device, dtype=torch.bfloat16)
        elif hasattr(
                batch,
                'preprocessed_image') and batch.preprocessed_image is not None:
            if isinstance(batch.preprocessed_image, torch.Tensor):
                if batch.preprocessed_image.dim() == 4:
                    video = batch.preprocessed_image.unsqueeze(2)
                elif batch.preprocessed_image.dim() == 5:
                    video = batch.preprocessed_image
        else:
            logger.info(
                "CosmosLatentPreparationStage - No video input sources found")

        if video is not None:
            num_cond_frames = video.size(2)
            if num_cond_frames >= num_frames:
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

                cfg = getattr(self.vae, "config", None)
                if (not bool(getattr(self.vae, "handles_latent_norm", False))
                        and cfg is not None and hasattr(cfg, 'latents_mean')
                        and hasattr(cfg, 'latents_std')):
                    latents_mean = torch.tensor(cfg.latents_mean).view(
                        1, cfg.z_dim, 1, 1, 1).to(device, dtype)
                    latents_std = torch.tensor(cfg.latents_std).view(
                        1, cfg.z_dim, 1, 1, 1).to(device, dtype)
                    init_latents = (init_latents - latents_mean
                                    ) / latents_std * self.scheduler.sigma_data

                conditioning_latents = init_latents
                self.vae.to("cpu")
        else:
            num_cond_latent_frames = 0

        if latents is None:
            seed = int(batch.seed if batch.seed is not None else 0)
            # Use arch-invariant RNG to match Cosmos2.5 reference sampling.
            latents_fp32 = self._arch_invariant_randn(shape,
                                                      seed=seed,
                                                      device=device,
                                                      dtype=torch.float32)
            latents = latents_fp32.to(torch.bfloat16)
        else:
            # If latents are supplied, keep compute dtype consistent with Cosmos sampling.
            latents = latents.to(device=device, dtype=torch.bfloat16)

        # Cosmos2.5 starts from unit Gaussian noise (no extra sigma_max scaling).

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

        # Cosmos 2.5 requires a spatial padding mask when concat_padding_mask=True
        padding_mask = latents.new_ones(batch_size, 1, latent_height,
                                        latent_width)

        batch.latents = latents
        batch.raw_latent_shape = latents.shape
        batch.conditioning_latents = conditioning_latents
        batch.cond_indicator = cond_indicator
        batch.uncond_indicator = uncond_indicator
        batch.cond_mask = cond_mask
        batch.uncond_mask = uncond_mask
        batch.padding_mask = padding_mask
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
