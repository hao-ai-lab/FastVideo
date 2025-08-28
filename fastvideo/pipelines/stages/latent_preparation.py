# SPDX-License-Identifier: Apache-2.0
"""
Latent preparation stage for diffusion pipelines.
"""

import torch
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

        # Calculate Cosmos-specific dimensions to match Diffusers exactly
        # Based on empirical analysis of Diffusers output shapes
        # Diffusers: 93 frames -> 7 latent frames, 704x1280 -> 88x160
        
        # For spatial: Need 704->88 and 1280->160
        # 704/88 = 8, 1280/160 = 8, so spatial_scale = 8
        vae_scale_factor_spatial = 8  
        
        # For temporal: Use the same scale factor as diffusers Cosmos pipeline
        # Diffusers uses vae_scale_factor_temporal = 4 as default
        # For 21 frames: (21-1)//4+1 = 20//4+1 = 5+1 = 6 latent frames (matches diffusers)
        vae_scale_factor_temporal = 4
        
        # Also check if height needs different scaling
        # 704 -> 88: 704/8 = 88 ✓
        # But maybe height uses different factor? 704/90 = 7.82 (not integer)
        # Let's use 704/88 = 8 exactly
        latent_height = height // 8  # Force to match Diffusers: 704//8 = 88
        latent_width = width // vae_scale_factor_spatial
        
        # Use same formula as diffusers cosmos pipeline
        num_latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1
        
        logger.info(f"CosmosLatentPreparationStage - input frames: {num_frames}, latent frames: {num_latent_frames}")
        logger.info(f"CosmosLatentPreparationStage - VAE scale factors: temporal={vae_scale_factor_temporal}, spatial={vae_scale_factor_spatial}")
        logger.info(f"CosmosLatentPreparationStage - Frame calculation: {num_frames} -> {num_latent_frames} (using formula: ({num_frames}-1)//{vae_scale_factor_temporal}+1)")
        logger.info(f"CosmosLatentPreparationStage - Dimensions: height={height} -> {latent_height}, width={width} -> {latent_width}")
        # latent_height and latent_width already calculated above
        
        # Cosmos transformer expects in_channels - 1 for the latent channels
        num_channels_latents = self.transformer.config.in_channels - 1
        logger.info(f"CosmosLatentPreparationStage - Final shape calculation: ({batch_size}, {num_channels_latents}, {num_latent_frames}, {latent_height}, {latent_width})")
        
        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)

        logger.info(f"CosmosLatentPreparationStage - Target shape: {shape}")
        
        # Debug: Double-check the calculation manually
        expected_frames = (num_frames - 1) // vae_scale_factor_temporal + 1
        expected_height = height // 8  
        expected_width = width // vae_scale_factor_spatial
        logger.info(f"CosmosLatentPreparationStage - Manual check: frames={expected_frames}, height={expected_height}, width={expected_width}")

        # Handle video input processing like diffusers
        init_latents = None
        conditioning_latents = None
        
        # Process input video if provided (video-to-world generation)
        # Check multiple possible sources for video input
        video = None
        logger.info(f"CosmosLatentPreparationStage - Checking for video inputs:")
        logger.info(f"  batch.video: {getattr(batch, 'video', 'Not found')}")
        logger.info(f"  batch.pil_image: {getattr(batch, 'pil_image', 'Not found')}")
        logger.info(f"  batch.preprocessed_image: {getattr(batch, 'preprocessed_image', 'Not found')}")
        
        if hasattr(batch, 'video') and batch.video is not None:
            video = batch.video
            logger.info("CosmosLatentPreparationStage - Using batch.video")
        elif hasattr(batch, 'pil_image') and batch.pil_image is not None:
            logger.info(f"CosmosLatentPreparationStage - Found pil_image of type: {type(batch.pil_image)}")
            # Convert single image to video format if needed
            if isinstance(batch.pil_image, torch.Tensor):
                logger.info(f"CosmosLatentPreparationStage - pil_image tensor shape: {batch.pil_image.shape}")
                if batch.pil_image.dim() == 4:  # [B, C, H, W] -> [B, C, T, H, W]
                    video = batch.pil_image.unsqueeze(2)
                    logger.info(f"CosmosLatentPreparationStage - Converted 4D to 5D tensor: {video.shape}")
                elif batch.pil_image.dim() == 5:  # Already [B, C, T, H, W]
                    video = batch.pil_image
                    logger.info(f"CosmosLatentPreparationStage - Using 5D tensor as-is: {video.shape}")
            else:
                logger.info("CosmosLatentPreparationStage - pil_image is not a tensor, needs preprocessing")
                # Following diffusers approach for image-to-video preprocessing
                # Convert PIL image to tensor and add temporal dimension
                import torchvision.transforms as transforms
                
                # Create transform pipeline similar to diffusers VideoProcessor
                transform = transforms.Compose([
                    transforms.Resize((height, width), antialias=True),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
                ])
                
                # Apply transform to get [C, H, W] tensor
                image_tensor = transform(batch.pil_image)
                logger.info(f"CosmosLatentPreparationStage - Transformed PIL to tensor: {image_tensor.shape}")
                
                # Add batch dimension: [C, H, W] -> [B, C, H, W]
                image_tensor = image_tensor.unsqueeze(0)
                
                # Add time dimension like diffusers: [B, C, H, W] -> [B, C, T, H, W]
                video = image_tensor.unsqueeze(2)  # Add time dim at position 2
                logger.info(f"CosmosLatentPreparationStage - Added batch and time dims: {video.shape}")
                
                # Move to correct device and ensure compatible dtype for VAE
                # Use VAE's parameter dtype to avoid dtype mismatches
                if self.vae is not None:
                    vae_dtype = next(self.vae.parameters()).dtype
                else:
                    vae_dtype = dtype
                video = video.to(device=device, dtype=vae_dtype)
                logger.info(f"CosmosLatentPreparationStage - Video tensor device: {video.device}, dtype: {video.dtype}")
        elif hasattr(batch, 'preprocessed_image') and batch.preprocessed_image is not None:
            logger.info(f"CosmosLatentPreparationStage - Found preprocessed_image of type: {type(batch.preprocessed_image)}")
            # Convert preprocessed image to video format
            if isinstance(batch.preprocessed_image, torch.Tensor):
                logger.info(f"CosmosLatentPreparationStage - preprocessed_image tensor shape: {batch.preprocessed_image.shape}")
                if batch.preprocessed_image.dim() == 4:  # [B, C, H, W] -> [B, C, T, H, W]
                    video = batch.preprocessed_image.unsqueeze(2)
                    logger.info(f"CosmosLatentPreparationStage - Converted 4D to 5D tensor: {video.shape}")
                elif batch.preprocessed_image.dim() == 5:  # Already [B, C, T, H, W]
                    video = batch.preprocessed_image
                    logger.info(f"CosmosLatentPreparationStage - Using 5D tensor as-is: {video.shape}")
        else:
            logger.info("CosmosLatentPreparationStage - No video input sources found")
        
        if video is not None:
            num_cond_frames = video.size(2)
            
            logger.info(f"CosmosLatentPreparationStage - Number of conditioning frames: {num_cond_frames}")
            
            if num_cond_frames >= num_frames:
                # Take the last `num_frames` frames for conditioning
                num_cond_latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1
                video = video[:, :, -num_frames:]
                logger.info(f"CosmosLatentPreparationStage - Using last {num_frames} frames from {num_cond_frames} conditioning frames")
            else:
                num_cond_latent_frames = (num_cond_frames - 1) // vae_scale_factor_temporal + 1
                num_padding_frames = num_frames - num_cond_frames
                last_frame = video[:, :, -1:]
                padding = last_frame.repeat(1, 1, num_padding_frames, 1, 1)
                video = torch.cat([video, padding], dim=2)
                logger.info(f"CosmosLatentPreparationStage - Padding {num_cond_frames} conditioning frames with {num_padding_frames} repeated frames")
            
            # Encode video through VAE like diffusers does
            if self.vae is not None:
                # Move VAE to correct device before encoding
                self.vae = self.vae.to(device)
                if isinstance(generator, list):
                    init_latents = []
                    for i in range(batch_size):
                        vae_output = self.vae.encode(video[i].unsqueeze(0))
                        logger.info(f"CosmosLatentPreparationStage - VAE output type: {type(vae_output)}, attributes: {dir(vae_output)}")
                        
                        # Handle different VAE output types
                        if hasattr(vae_output, 'latent_dist'):
                            init_latents.append(vae_output.latent_dist.sample(generator[i] if i < len(generator) else None))
                        elif hasattr(vae_output, 'latents'):
                            init_latents.append(vae_output.latents)
                        elif hasattr(vae_output, 'sample'):
                            init_latents.append(vae_output.sample(generator[i] if i < len(generator) else None))
                        elif isinstance(vae_output, torch.Tensor):
                            # Direct tensor output
                            init_latents.append(vae_output)
                        else:
                            # Try to get the first attribute that looks like latents
                            attrs = [attr for attr in dir(vae_output) if not attr.startswith('_')]
                            logger.info(f"CosmosLatentPreparationStage - Available attributes: {attrs}")
                            raise AttributeError(f"Could not access latents from VAE output. Available attributes: {attrs}")
                else:
                    init_latents_list = []
                    for vid in video:
                        vae_output = self.vae.encode(vid.unsqueeze(0))
                        logger.info(f"CosmosLatentPreparationStage - VAE output type: {type(vae_output)}, attributes: {dir(vae_output)}")
                        
                        # Handle different VAE output types
                        if hasattr(vae_output, 'latent_dist'):
                            init_latents_list.append(vae_output.latent_dist.sample(generator))
                        elif hasattr(vae_output, 'latents'):
                            init_latents_list.append(vae_output.latents)
                        elif hasattr(vae_output, 'sample'):
                            init_latents_list.append(vae_output.sample(generator))
                        elif isinstance(vae_output, torch.Tensor):
                            # Direct tensor output
                            init_latents_list.append(vae_output)
                        else:
                            # Try to get the first attribute that looks like latents
                            attrs = [attr for attr in dir(vae_output) if not attr.startswith('_')]
                            logger.info(f"CosmosLatentPreparationStage - Available attributes: {attrs}")
                            raise AttributeError(f"Could not access latents from VAE output. Available attributes: {attrs}")
                    init_latents = init_latents_list
                
                init_latents = torch.cat(init_latents, dim=0).to(dtype)
                
                # Apply latent normalization like diffusers
                if hasattr(self.vae.config, 'latents_mean') and hasattr(self.vae.config, 'latents_std'):
                    latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(device, dtype)
                    latents_std = torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(device, dtype)
                    init_latents = (init_latents - latents_mean) / latents_std * self.scheduler.sigma_data
                
                conditioning_latents = init_latents
                
                # Offload VAE to CPU after encoding to save memory
                self.vae.to("cpu")
        else:
            num_cond_latent_frames = 0
            logger.info("CosmosLatentPreparationStage - No conditioning frames detected (no video input)")
        
        # Generate or use provided latents
        if latents is None:
            latents = randn_tensor(shape,
                                 generator=generator,
                                 device=device,
                                 dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        # Scale latents by sigma_max (Cosmos-specific) - exactly like diffusers
        latents = latents * self.scheduler.sigma_max

        # Create conditioning masks (for video-to-world generation)
        padding_shape = (batch_size, 1, num_latent_frames, latent_height, latent_width)
        ones_padding = latents.new_ones(padding_shape)
        zeros_padding = latents.new_zeros(padding_shape)

        # Create conditioning indicators based on actual conditioning frames
        cond_indicator = latents.new_zeros(1, 1, latents.size(2), 1, 1)
        cond_indicator[:, :, :num_cond_latent_frames] = 1.0
        cond_mask = cond_indicator * ones_padding + (1 - cond_indicator) * zeros_padding

        # For classifier-free guidance
        uncond_indicator = None
        uncond_mask = None
        if batch.do_classifier_free_guidance:
            uncond_indicator = latents.new_zeros(1, 1, latents.size(2), 1, 1)
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
        batch.conditioning_latents = conditioning_latents
        batch.cond_indicator = cond_indicator
        batch.uncond_indicator = uncond_indicator
        batch.cond_mask = cond_mask
        batch.uncond_mask = uncond_mask
        
        # Final verification that shape is correct
        logger.info(f"CosmosLatentPreparationStage - FINAL latents shape: {latents.shape}")
        # Compare with Diffusers expected shape for our dimensions
        # For 21 frames with temporal_scale=4: (21-1)//4+1 = 6 latent frames
        diffusers_expected = torch.Size([1, 16, 6, 88, 160])
        if latents.shape != diffusers_expected:
            logger.warning(f"CosmosLatentPreparationStage - Shape differs from Diffusers: Expected {diffusers_expected}, got {latents.shape}")
            logger.info(f"CosmosLatentPreparationStage - This may be due to different input dimensions (height={height}, width={width})")
            logger.info(f"CosmosLatentPreparationStage - Debug values: batch_size={batch_size}, num_channels={num_channels_latents}, frames={num_latent_frames}, h={latent_height}, w={latent_width}")
            logger.info(f"CosmosLatentPreparationStage - Input values: num_frames={num_frames}, height={height}, width={width}")

        logger.info(f"CosmosLatentPreparationStage - final latents shape: {latents.shape}")
        logger.info(f"CosmosLatentPreparationStage - conditioning frames: {num_cond_latent_frames}/{num_latent_frames}")
        logger.info(f"CosmosLatentPreparationStage - cond_mask shape: {cond_mask.shape}")
        if conditioning_latents is not None:
            logger.info(f"CosmosLatentPreparationStage - conditioning_latents shape: {conditioning_latents.shape}")

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
