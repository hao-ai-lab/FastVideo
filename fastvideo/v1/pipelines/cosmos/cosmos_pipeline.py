# SPDX-License-Identifier: Apache-2.0
"""
Cosmos video diffusion pipeline implementation.

This module contains an implementation of the Cosmos video diffusion pipeline
using the modular pipeline architecture.
"""

import os
import numpy as np
import torch

from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.v1.pipelines.stages import (ConditioningStage, DecodingStage,
                                           DenoisingStage, InputValidationStage,
                                           LatentPreparationStage,
                                           TextEncodingStage,
                                           TimestepPreparationStage)
from fastvideo.v1.pipelines.stages.base import PipelineStage

logger = init_logger(__name__)


class VideoPreprocessingStage(PipelineStage):
    """
    Video preprocessing stage for Cosmos pipeline.
    
    This stage handles preprocessing of both image and video inputs,
    similar to the video_processor in the diffuser's implementation.
    """
    
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        # TODO: Implement video preprocessing logic
        # This should handle both image->video and video->video preprocessing
        # Similar to diffuser's video_processor.preprocess() and preprocess_video()
        
    def forward(self, batch, fastvideo_args):
        """
        Preprocess video/image inputs.
        
        Args:
            batch: The current batch information
            fastvideo_args: The inference arguments
            
        Returns:
            The batch with preprocessed video data
        """
        # TODO: Implement video preprocessing
        # This should:
        # 1. Handle both image and video inputs
        # 2. Resize/crop to target dimensions
        # 3. Convert to proper format for VAE encoding
        # 4. Apply any necessary normalization
        
        return batch


class CosmosLatentPreparationStage(LatentPreparationStage):
    """
    Specialized latent preparation stage for Cosmos.
    
    This extends the base LatentPreparationStage to handle Cosmos-specific
    conditioning logic including masks and indicators for video2world generation.
    """
    
    def __init__(self, scheduler, transformer, vae):
        super().__init__(scheduler, transformer)
        self.vae = vae
    
    def forward(self, batch, fastvideo_args):
        """
        Prepare latents with Cosmos-specific conditioning.
        
        This implements the complex latent preparation logic from the diffuser's
        implementation for video2world generation, including:
        - Video encoding to conditioning latents
        - Conditioning indicators (which frames have input)
        - Conditioning masks (for blending input/generated frames)  
        - Proper scaling and normalization
        """
        from diffusers.utils.torch_utils import randn_tensor
        from fastvideo.v1.distributed import get_local_torch_device
        
        # Get basic parameters
        device = get_local_torch_device()
        
        # Determine batch size
        if isinstance(batch.prompt, list):
            batch_size = len(batch.prompt)
        elif batch.prompt is not None:
            batch_size = 1
        else:
            batch_size = batch.prompt_embeds[0].shape[0]
        
        batch_size *= batch.num_videos_per_prompt
        
        # Calculate latent dimensions
        num_frames = batch.num_frames or 93  # Default for Cosmos
        height = batch.height or 704  # Default for Cosmos
        width = batch.width or 1280  # Default for Cosmos
        
        # Get video tensor (should be preprocessed by VideoPreprocessingStage)
        video = batch.extra.get('preprocessed_video') or batch.preprocessed_image
        if video is None:
            # Check if we have an image_path in the batch (required for Video2World)
            image_path = batch.image_path
            if image_path and os.path.exists(image_path):
                # Load and preprocess real image
                from PIL import Image
                import torch.nn.functional as F
                
                # Load image
                pil_image = Image.open(image_path).convert('RGB')
                
                # Convert to tensor and normalize
                image_tensor = torch.tensor(np.array(pil_image)).float().permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
                image_tensor = image_tensor / 255.0 * 2.0 - 1.0  # Normalize to [-1, 1]
                
                # Resize to target dimensions
                image_tensor = F.interpolate(image_tensor, size=(height, width), mode='bilinear', align_corners=False)
                
                # Add time dimension - keep as 1 frame for now (diffuser approach)
                video = image_tensor.unsqueeze(2)  # [B, C, 1, H, W]
                # Don't repeat here - let the conditioning logic handle frame expansion
                # Convert to correct dtype and device
                vae_dtype = next(self.vae.parameters()).dtype
                video = video.to(device=device, dtype=vae_dtype)
                
                logger.info(f"✅ Loaded and preprocessed image from {image_path} with shape {video.shape} (expanded to {num_frames} frames before VAE encoding)")
            else:
                # For Video2World, we should normally require an image_path
                # This fallback is only for development/testing purposes
                if image_path is None:
                    logger.error("❌ Video2World pipeline requires image_path parameter. Please provide an input image.")
                    logger.error("   Example: generator.generate_video(prompt='...', image_path='your_image.jpg')")
                else:
                    logger.error(f"❌ Image file not found: {image_path}")
                
                logger.warning("🔧 Creating test image for development purposes only...")
                
                # Create a simple test image (for development only)
                vae_dtype = next(self.vae.parameters()).dtype
                test_image = torch.zeros(height, width, 3, dtype=vae_dtype)
                for i in range(height):
                    for j in range(width):
                        test_image[i, j, 0] = i / height  # Red gradient
                        test_image[i, j, 1] = j / width   # Green gradient 
                        test_image[i, j, 2] = 0.5         # Blue constant
                test_image = test_image.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
                
                # Simple preprocessing
                import torch.nn.functional as F
                test_image = F.interpolate(test_image, size=(height, width), mode='bilinear', align_corners=False)
                test_image = test_image * 2.0 - 1.0  # Normalize to [-1, 1]
                
                # Add time dimension - keep as 1 frame for now (diffuser approach)
                video = test_image.unsqueeze(2)  # [B, C, 1, H, W]
                # Don't repeat here - let the conditioning logic handle frame expansion
                video = video.to(device=device)
                
                logger.warning(f"🔧 Created test video with shape {video.shape} - Use image_path for real usage!")
        
        # Calculate temporal compression
        vae_scale_factor_temporal = getattr(self.vae, 'temperal_downsample', [2, 2])
        if isinstance(vae_scale_factor_temporal, list):
            vae_scale_factor_temporal = 2 ** sum(vae_scale_factor_temporal)
        else:
            vae_scale_factor_temporal = 4  # default
            
        # Calculate spatial compression  
        vae_scale_factor_spatial = getattr(self.vae, 'spatial_compression_ratio', 8)
        
        num_latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1
        latent_height = height // vae_scale_factor_spatial
        latent_width = width // vae_scale_factor_spatial
        
        # Handle conditioning frames (matching diffuser's approach)
        num_cond_frames = video.size(2)
        logger.info(f"🔍 Input video has {num_cond_frames} frames, target is {num_frames} frames")
        
        if num_cond_frames >= num_frames:
            # Take the last `num_frames` frames for conditioning
            num_cond_latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1
            video = video[:, :, -num_frames:]
            logger.info(f"🔍 Using last {num_frames} frames, conditioning {num_cond_latent_frames} latent frames")
        else:
            # For image input: only condition the first frame, generate the rest
            num_cond_latent_frames = (num_cond_frames - 1) // vae_scale_factor_temporal + 1
            num_padding_frames = num_frames - num_cond_frames
            last_frame = video[:, :, -1:]
            padding = last_frame.repeat(1, 1, num_padding_frames, 1, 1)
            video = torch.cat([video, padding], dim=2)
            logger.info(f"🔍 Image input: {num_cond_frames} real frames + {num_padding_frames} padding frames")
            logger.info(f"🔍 Only conditioning {num_cond_latent_frames} latent frames (first frame only)")
        
        logger.info(f"🔍 Final video shape: {video.shape}, conditioning frames: {num_cond_latent_frames}")
        
        # Encode the ENTIRE video (including padding) to match diffusers exactly
        # This ensures we encode the same number of frames as diffusers
        # Ensure VAE is ready for encoding (initialize cache if needed)
        if hasattr(self.vae, 'use_feature_cache') and self.vae.use_feature_cache:
            if not hasattr(self.vae, '_enc_feat_map'):
                self.vae.clear_cache()
        
        generator = batch.generator
        if isinstance(generator, list):
            init_latents = []
            for i in range(batch_size):
                encoder_output = self.vae.encode(video[i].unsqueeze(0))
                latent = self._retrieve_latents(encoder_output, generator[i])
                init_latents.append(latent)
            conditioning_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = []
            for vid in video:
                encoder_output = self.vae.encode(vid.unsqueeze(0))
                latent = self._retrieve_latents(encoder_output, generator)
                init_latents.append(latent)
            conditioning_latents = torch.cat(init_latents, dim=0)
        
        # Apply VAE scaling and normalization
        dtype = batch.prompt_embeds[0].dtype
        conditioning_latents = conditioning_latents.to(dtype)
        
        # Get VAE config for normalization
        # Use sigma_data from scheduler config, matching diffusers exactly
        if hasattr(self.vae.config, 'latents_mean') and self.vae.config.latents_mean is not None:
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(device, dtype)
            latents_std = torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(device, dtype)
            conditioning_latents = (conditioning_latents - latents_mean) / latents_std * self.scheduler.config.sigma_data
        else:
            conditioning_latents = conditioning_latents * self.scheduler.config.sigma_data
        
        # Generate random latents for generation
        # For Cosmos Video2World: transformer expects 16 channels in hidden_states
        # The conditioning is passed separately via condition_mask parameter (not concatenated)
        num_channels_latents = self.transformer.config.in_channels - 1
        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        
        latents = batch.latents
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)
        
        # Scale latents by scheduler's sigma_max (default to 80.0 for Cosmos)
        sigma_max = getattr(self.scheduler.config, 'sigma_max', 80.0)
        latents = latents * sigma_max
        
        # Create conditioning indicators and masks
        # IMPORTANT: condition_mask should be in latent space dimensions (like diffusers)
        padding_shape = (batch_size, 1, num_latent_frames, latent_height, latent_width)
        ones_padding = latents.new_ones(padding_shape)
        zeros_padding = latents.new_zeros(padding_shape)
        
        # Conditioning indicator: 1 for frames with input, 0 for generated frames
        cond_indicator = latents.new_zeros(1, 1, latents.size(2), 1, 1)
        cond_indicator[:, :, :num_cond_latent_frames] = 1.0
        cond_mask = cond_indicator * ones_padding + (1 - cond_indicator) * zeros_padding
        
        # Debug: Check conditioning setup
        logger.info(f"🔍 Latent shape: {latents.shape} (T={latents.size(2)} frames)")
        logger.info(f"🔍 Conditioning indicator shape: {cond_indicator.shape}")
        logger.info(f"🔍 Conditioning first {num_cond_latent_frames} latent frames out of {latents.size(2)}")
        logger.info(f"🔍 Cond indicator values: {cond_indicator.squeeze().tolist()}")
        
        # Log VAE encoding info for comparison test (matching diffusers format)
        logger.info(f"[VAE] Input video stats: shape: {list(video.shape)}, mean: {video.float().mean().item():.4f}, std: {video.float().std().item():.4f}")
        logger.info(f"[VAE] Encoded latent shape: {list(conditioning_latents.shape)}")
        logger.info(f"[VAE] Encoded latent stats: mean: {conditioning_latents.float().mean().item():.4f}, std: {conditioning_latents.float().std().item():.4f}")
        logger.info(f"[VAE] Input video range: [{video.float().min().item():.4f}, {video.float().max().item():.4f}]")
        logger.info(f"[VAE] Input video sum: {video.float().sum().item():.4f}")
        logger.info(f"[VAE] Input video abs_sum: {video.float().abs().sum().item():.4f}")
        logger.info(f"[VAE] Input video abs_max: {video.float().abs().max().item():.4f}")
        logger.info(f"[VAE] Input video norm: {video.float().norm().item():.4f}")
        logger.info(f"[VAE] Encoded latent range: [{conditioning_latents.float().min().item():.4f}, {conditioning_latents.float().max().item():.4f}]")
        logger.info(f"[VAE] Encoded latent sum: {conditioning_latents.float().sum().item():.4f}")
        logger.info(f"[VAE] Encoded latent abs_sum: {conditioning_latents.float().abs().sum().item():.4f}")
        logger.info(f"[VAE] Encoded latent abs_max: {conditioning_latents.float().abs().max().item():.4f}")
        logger.info(f"[VAE] Encoded latent norm: {conditioning_latents.float().norm().item():.4f}")
        logger.info(f"[VAE] Conditioning first {num_cond_latent_frames} latent frames out of {latents.size(2)} total")
        logger.info(f"[VAE] Conditioning indicator: {cond_indicator.squeeze().tolist()}")
        
        # Unconditioning mask for classifier-free guidance
        uncond_indicator = None
        uncond_mask = None
        if batch.do_classifier_free_guidance:
            uncond_indicator = latents.new_zeros(1, 1, latents.size(2), 1, 1)
            uncond_indicator[:, :, :num_cond_latent_frames] = 1.0
            uncond_mask = uncond_indicator * ones_padding + (1 - uncond_indicator) * zeros_padding
        
        # Store all the conditioning information in the batch
        batch.latents = latents
        batch.raw_latent_shape = latents.shape  # Store the shape tuple for validation
        
        # Debug logging for channel verification
        logger.warning(f"Prepared latents shape: {latents.shape} ({latents.shape[1]} channels)")
        logger.info(f"✅ Using diffuser-style conditioning: 16 channels + separate condition_mask")
        
        # Store Cosmos-specific conditioning for our custom denoising stage
        batch.extra['conditioning_latents'] = conditioning_latents  
        batch.extra['condition_mask'] = cond_mask
        
        # Important: Do NOT set batch.image_latent - we use separate condition_mask instead
        
        # Store additional Cosmos-specific info in extra for future use
        batch.extra['cond_indicator'] = cond_indicator
        batch.extra['uncond_indicator'] = uncond_indicator
        batch.extra['uncond_mask'] = uncond_mask
        
        # Set unconditioning_latents for CFG (matching diffusers)
        if batch.do_classifier_free_guidance:
            batch.extra['unconditioning_latents'] = conditioning_latents
        else:
            batch.extra['unconditioning_latents'] = None
        
        # Debug: Log unconditioning_latents setup
        if batch.do_classifier_free_guidance:
            logger.info(f"🔍 CFG enabled: unconditioning_latents shape: {conditioning_latents.shape}")
        else:
            logger.info(f"🔍 CFG disabled: unconditioning_latents is None")
        
        return batch
    
    def _retrieve_latents(self, encoder_output, generator=None, sample_mode="sample"):
        """Retrieve latents from VAE encoder output."""
        if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
            return encoder_output.latent_dist.sample(generator)
        elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
            return encoder_output.latent_dist.mode()
        elif hasattr(encoder_output, "latents"):
            return encoder_output.latents
        elif hasattr(encoder_output, "mean"):
            return encoder_output.mean
        else:
            # Fallback: assume encoder_output is the latent tensor itself
            return encoder_output


class CosmosDenoisingStage(DenoisingStage):
    """
    Cosmos-specific denoising stage implementing proper conditioning logic from diffuser.
    """
    
    def forward(self, batch, fastvideo_args):
        """
        Run Cosmos-specific denoising with proper conditioning like the diffuser's implementation.
        
        This implements the complex conditioning logic including:
        - augment_sigma noise injection
        - Proper frame blending with cond_indicator
        - Separate conditional/unconditional paths for CFG
        - Correct padding_mask handling
        """
        from diffusers.utils.torch_utils import randn_tensor
        from fastvideo.v1.distributed import get_local_torch_device
        
        # Get Cosmos-specific conditioning from batch.extra
        conditioning_latents = batch.extra.get('conditioning_latents')
        condition_mask = batch.extra.get('condition_mask')  
        cond_indicator = batch.extra.get('cond_indicator')
        uncond_indicator = batch.extra.get('uncond_indicator')
        uncond_mask = batch.extra.get('uncond_mask')
        unconditioning_latents = batch.extra.get('unconditioning_latents')
        
        if conditioning_latents is None or cond_indicator is None:
            raise ValueError("Missing conditioning latents or indicators - check CosmosLatentPreparationStage")
        
        # Prepare extra step kwargs for scheduler
        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step,
            {"generator": batch.generator, "eta": batch.eta},
        )

        # Setup precision  
        target_dtype = torch.bfloat16
        autocast_enabled = True

        # Get timesteps and parameters
        timesteps = batch.timesteps
        if timesteps is None:
            raise ValueError("Timesteps must be provided")
        
        num_inference_steps = batch.num_inference_steps
        latents = batch.latents
        if latents is None:
            raise ValueError("Latents must be provided")
        prompt_embeds = batch.prompt_embeds[0]  # Take first embedding
        
        # Cosmos-specific parameters (matching diffuser's Cosmos2 implementation)
        height = batch.height or 704  # Default height if None
        width = batch.width or 1280   # Default width if None
        padding_mask = latents.new_zeros(1, 1, height, width, dtype=target_dtype)
        sigma_conditioning = torch.tensor(0.0001, dtype=torch.float32, device=latents.device)
        t_conditioning = (sigma_conditioning / (sigma_conditioning + 1)).to(latents.device)
        
        logger.info("🎬 Starting Cosmos2 denoising with diffuser-exact conditioning logic...")
        
        # Log denoising info for comparison test (matching diffusers format)
        logger.info(f"[DENOISE] Starting denoising loop with {num_inference_steps} steps")
        logger.info(f"[DENOISE] latents shape: {list(latents.shape)}")
        logger.info(f"[DENOISE] guidance_scale: {batch.guidance_scale}")
        
        # Debug scheduler setup
        logger.info(f"[DENOISE] Scheduler type: {type(self.scheduler)}")
        logger.info(f"[DENOISE] Scheduler has sigmas: {hasattr(self.scheduler, 'sigmas')}")
        if hasattr(self.scheduler, 'sigmas'):
            logger.info(f"[DENOISE] Scheduler sigmas shape: {self.scheduler.sigmas.shape if self.scheduler.sigmas is not None else 'None'}")
            logger.info(f"[DENOISE] Full sigma schedule: {self.scheduler.sigmas.tolist() if self.scheduler.sigmas is not None else 'None'}")
        logger.info(f"[DENOISE] Timesteps shape: {timesteps.shape}")
        logger.info(f"[DENOISE] Full timesteps schedule: {timesteps.tolist()}")
        
        # Denoising loop matching diffuser's Cosmos2VideoToWorld exactly
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Ensure we use the proper sigma values from scheduler (matching diffusers exactly)
                if hasattr(self.scheduler, 'sigmas') and self.scheduler.sigmas is not None:
                    current_sigma = self.scheduler.sigmas[i]
                    logger.info(f"🔍 Step {i}: Using sigma from scheduler: {current_sigma.item():.6f}")
                else:
                    # Fallback: calculate sigma from timestep (should not happen with proper scheduler setup)
                    logger.warning(f"🔍 Step {i}: Scheduler missing sigmas attribute, using fallback calculation")
                    current_sigma = t
                
                current_sigma = current_sigma.to(latents.device)
                
                # Cosmos2 conditioning coefficients (from diffuser)
                current_t = current_sigma / (current_sigma + 1)
                c_in = 1 - current_t
                c_skip = 1 - current_t  
                c_out = -current_t
                
                # Log sigma and coefficient info for first step only
                if i == 0:
                    logger.info(f"[DENOISE] Step {i}: current_sigma={current_sigma.item():.6f}, current_t={current_t.item():.6f}")
                    logger.info(f"[DENOISE] Step {i}: c_in={c_in.item():.6f}, c_skip={c_skip.item():.6f}, c_out={c_out.item():.6f}")
                timestep = current_t.view(1, 1, 1, 1, 1).expand(
                    latents.size(0), -1, latents.size(2), -1, -1
                )  # [B, 1, T, 1, 1]
                
                # Prepare conditional latent (matching diffuser exactly)
                cond_latent = latents * c_in
                
                # Apply conditioning exactly like diffusers
                cond_latent = cond_indicator * conditioning_latents + (1 - cond_indicator) * cond_latent
                cond_latent = cond_latent.to(target_dtype)
                cond_timestep = cond_indicator * t_conditioning + (1 - cond_indicator) * timestep
                cond_timestep = cond_timestep.to(target_dtype)
                
                # Debug: Check what we're sending to transformer
                if i == 0:  # Only log first step
                    logger.info(f"🔍 Step {i}: cond_latent shape: {cond_latent.shape}")
                    logger.info(f"🔍 Step {i}: cond_timestep shape: {cond_timestep.shape}")
                    if condition_mask is not None:
                        logger.info(f"🔍 Step {i}: condition_mask shape: {condition_mask.shape}")
                    logger.info(f"🔍 Step {i}: conditioning_latents mean: {conditioning_latents.mean().item():.4f}")
                    logger.info(f"🔍 Step {i}: latents (noise) mean: {latents.mean().item():.4f}")
                    logger.info(f"🔍 Step {i}: cond_latent mean: {cond_latent.mean().item():.4f}")
                    
                    # Log detailed conditioning analysis
                    logger.info(f"🔍 Step {i}: conditioning_latents mean: {conditioning_latents.float().mean().item():.4f}")
                    logger.info(f"🔍 Step {i}: latents (noise) mean: {latents.float().mean().item():.4f}")
                    logger.info(f"🔍 Step {i}: cond_latent mean: {cond_latent.float().mean().item():.4f}")
                    
                    # Call Cosmos transformer (16 channels only!)
                with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled):
                    from fastvideo.v1.forward_context import set_forward_context
                    
                    with set_forward_context(
                            current_timestep=i,
                            attn_metadata=None,
                            forward_batch=batch,
                    ):
                        noise_pred = self.transformer(
                            hidden_states=cond_latent,  # 16 channels, not 17!
                            timestep=cond_timestep,
                            encoder_hidden_states=prompt_embeds,
                            fps=batch.fps or 16,
                            condition_mask=condition_mask,  # Separate conditioning
                            padding_mask=padding_mask,
                            return_dict=False,
                        )[0]
                
                # EXACT Cosmos2 implementation from diffuser
                raw_noise_pred = noise_pred.clone()  # Store original for debugging
                noise_pred = (c_skip * latents + c_out * noise_pred.float()).to(target_dtype)
                noise_pred = cond_indicator * conditioning_latents + (1 - cond_indicator) * noise_pred
                
                # Debug: Check noise prediction
                if i == 0:  # Only log first step
                    logger.info(f"🔍 Step {i}: raw transformer output mean: {raw_noise_pred.float().mean().item():.4f}")
                    logger.info(f"🔍 Step {i}: after coefficients mean: {(c_skip * latents + c_out * raw_noise_pred.float()).float().mean().item():.4f}")
                    logger.info(f"🔍 Step {i}: final noise_pred mean: {noise_pred.float().mean().item():.4f}")
                    logger.info(f"🔍 Step {i}: c_in={c_in.item():.4f}, c_skip={c_skip.item():.4f}, c_out={c_out.item():.4f}")
                    
                    # Log noise prediction info for comparison test (matching diffusers format)
                    logger.info(f"[DENOISE] noise_pred shape: {list(noise_pred.shape)}")
                    logger.info(f"[DENOISE] noise_pred stats: mean: {noise_pred.float().mean().item():.4f}, std: {noise_pred.float().std().item():.4f}")
                    logger.info(f"[DENOISE] noise_pred range: [{noise_pred.float().min().item():.4f}, {noise_pred.float().max().item():.4f}]")
                    logger.info(f"[DENOISE] noise_pred sum: {noise_pred.float().sum().item():.4f}")
                    logger.info(f"[DENOISE] noise_pred abs_sum: {noise_pred.float().abs().sum().item():.4f}")
                    logger.info(f"[DENOISE] noise_pred abs_max: {noise_pred.float().abs().max().item():.4f}")
                    logger.info(f"[DENOISE] noise_pred norm: {noise_pred.float().norm().item():.4f}")
                    logger.info(f"[DENOISE] c_in={c_in.item():.4f}, c_skip={c_skip.item():.4f}, c_out={c_out.item():.4f}")
                    logger.info(f"[DENOISE] latents per frame means: {[latents[0, :, f].float().mean().item() for f in range(latents.shape[2])]}")
                    logger.info(f"[DENOISE] latents range: [{latents.float().min().item():.4f}, {latents.float().max().item():.4f}]")
                    logger.info(f"[DENOISE] latents mean: {latents.float().mean().item():.4f}")
                    logger.info(f"[DENOISE] latents std: {latents.float().std().item():.4f}")
                    logger.info(f"[DENOISE] latents sum: {latents.float().sum().item():.4f}")
                    logger.info(f"[DENOISE] latents abs_sum: {latents.float().abs().sum().item():.4f}")
                    logger.info(f"[DENOISE] latents abs_max: {latents.float().abs().max().item():.4f}")
                    logger.info(f"[DENOISE] latents norm: {latents.float().norm().item():.4f}")
                    
                    # Log guidance scale for comparison
                    logger.info(f"[DENOISE] guidance_scale: {batch.guidance_scale}")
                    
                    # Log detailed conditioning analysis (matching diffusers format)
                    logger.info(f"[CONDITIONING] Step {i}: cond_latent mean: {cond_latent.float().mean().item():.4f}")
                    logger.info(f"[CONDITIONING] Step {i}: cond_latent range: [{cond_latent.float().min().item():.4f}, {cond_latent.float().max().item():.4f}]")
                    logger.info(f"[CONDITIONING] Step {i}: cond_latent std: {cond_latent.float().std().item():.4f}")
                    logger.info(f"[CONDITIONING] Step {i}: cond_latent abs_max: {cond_latent.float().abs().max().item():.4f}")
                    logger.info(f"[CONDITIONING] Step {i}: raw transformer output mean: {raw_noise_pred.float().mean().item():.4f}")
                    logger.info(f"[CONDITIONING] Step {i}: raw transformer output range: [{raw_noise_pred.float().min().item():.4f}, {raw_noise_pred.float().max().item():.4f}]")
                    logger.info(f"[CONDITIONING] Step {i}: raw transformer output std: {raw_noise_pred.float().std().item():.4f}")
                    logger.info(f"[CONDITIONING] Step {i}: after coefficients mean: {(c_skip * latents + c_out * raw_noise_pred.float()).float().mean().item():.4f}")
                    logger.info(f"[CONDITIONING] Step {i}: after coefficients range: [{(c_skip * latents + c_out * raw_noise_pred.float()).float().min().item():.4f}, {(c_skip * latents + c_out * raw_noise_pred.float()).float().max().item():.4f}]")
                    logger.info(f"[CONDITIONING] Step {i}: final noise_pred mean: {noise_pred.float().mean().item():.4f}")
                    logger.info(f"[CONDITIONING] Step {i}: final noise_pred range: [{noise_pred.float().min().item():.4f}, {noise_pred.float().max().item():.4f}]")
                    logger.info(f"[CONDITIONING] Step {i}: final noise_pred std: {noise_pred.float().std().item():.4f}")
                    
                    # Log per-frame conditioning analysis
                    first_frame_cond = cond_latent[0, :, 0].float().mean().item()
                    other_frames_cond = cond_latent[0, :, 1:].float().mean().item()
                    logger.info(f"[CONDITIONING] Step {i}: First frame cond_latent mean: {first_frame_cond:.4f}")
                    logger.info(f"[CONDITIONING] Step {i}: Other frames cond_latent mean: {other_frames_cond:.4f}")
                    logger.info(f"[CONDITIONING] Step {i}: Conditioning difference: {abs(first_frame_cond - other_frames_cond):.4f}")
                    
                    # Log step-specific latents and noise_pred means
                    logger.info(f"[CONDITIONING] Step {i}: latents mean: {latents.float().mean().item():.6f}, noise_pred mean: {noise_pred.float().mean().item():.6f}")
                    logger.info(f"[CONDITIONING] Step {i}: current_sigma: {current_sigma.item():.6f}")
                
                # Handle classifier-free guidance if enabled  
                if batch.do_classifier_free_guidance and uncond_indicator is not None:
                    negative_prompt_embeds = batch.negative_prompt_embeds[0] if batch.negative_prompt_embeds else prompt_embeds
                    
                    # Prepare unconditional path (matching diffuser's approach)
                    uncond_latent = latents * c_in
                    uncond_latent = uncond_indicator * unconditioning_latents + (1 - uncond_indicator) * uncond_latent
                    uncond_latent = uncond_latent.to(target_dtype)
                    uncond_timestep = uncond_indicator * t_conditioning + (1 - uncond_indicator) * timestep
                    uncond_timestep = uncond_timestep.to(target_dtype)
                    
                    # Debug: Log unconditional path setup
                    if i == 0:  # Only log first step
                        logger.info(f"🔍 Step {i}: uncond_latent shape: {uncond_latent.shape}")
                        logger.info(f"🔍 Step {i}: uncond_latent mean: {uncond_latent.mean().item():.4f}")
                        if unconditioning_latents is not None:
                            logger.info(f"🔍 Step {i}: unconditioning_latents mean: {unconditioning_latents.mean().item():.4f}")
                        else:
                            logger.info(f"🔍 Step {i}: unconditioning_latents is None")
                        logger.info(f"🔍 Step {i}: uncond_indicator values: {uncond_indicator.squeeze().tolist()}")
                    
                    with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled):
                        with set_forward_context(
                                current_timestep=i,
                                attn_metadata=None,
                                forward_batch=batch,
                        ):
                            noise_pred_uncond = self.transformer(
                                hidden_states=uncond_latent,
                                timestep=uncond_timestep,
                                encoder_hidden_states=negative_prompt_embeds,
                                fps=batch.fps or 16,
                                condition_mask=uncond_mask,
                                padding_mask=padding_mask,
                                return_dict=False,
                            )[0]
                    
                    # Apply diffuser's post-processing for unconditional
                    noise_pred_uncond = (c_skip * latents + c_out * noise_pred_uncond.float()).to(target_dtype)
                    noise_pred_uncond = uncond_indicator * unconditioning_latents + (1 - uncond_indicator) * noise_pred_uncond
                    
                    # Apply classifier-free guidance
                    noise_pred = noise_pred + batch.guidance_scale * (noise_pred - noise_pred_uncond)
                
                # EXACT Cosmos2 scheduler step from diffuser
                logger.info(f"🔍 Step {i}: latents mean: {latents.mean().item():.6f}, noise_pred mean: {noise_pred.mean().item():.6f}")
                logger.info(f"🔍 Step {i}: current_sigma: {current_sigma.item():.6f}")
                noise_pred_final = (latents - noise_pred) / current_sigma
                logger.info(f"🔍 Step {i}: noise_pred_final mean: {noise_pred_final.mean().item():.6f}")
                latents = self.scheduler.step(noise_pred_final, t.to(latents.device), latents, return_dict=False)[0]
                
                # Debug: Check final latents
                if i == 0:  # Only log first step
                    logger.info(f"🔍 Step {i}: noise_pred_final mean: {noise_pred_final.float().mean().item():.4f}")
                    logger.info(f"🔍 Step {i}: final latents mean: {latents.float().mean().item():.4f}")
                    logger.info(f"🔍 Step {i}: latents per frame means: {[latents[0, :, f].float().mean().item() for f in range(latents.shape[2])]}")
                    
                    # Check final conditioning state
                    first_frame_final = latents[0, :, 0].float().mean().item()
                    other_frames_final = latents[0, :, 1:].float().mean().item()
                    logger.info(f"🔍 Step {i}: First frame final mean: {first_frame_final:.4f}")
                    logger.info(f"🔍 Step {i}: Other frames final mean: {other_frames_final:.4f}")
                    logger.info(f"🔍 Step {i}: Final conditioning difference: {abs(first_frame_final - other_frames_final):.4f}")
                    
                    # Log final step values (matching diffusers format)
                    logger.info(f"[CONDITIONING] Step {i}: noise_pred_final mean: {noise_pred_final.float().mean().item():.4f}")
                    logger.info(f"[CONDITIONING] Step {i}: final latents mean: {latents.float().mean().item():.4f}")
                    logger.info(f"[CONDITIONING] Step {i}: First frame final mean: {first_frame_final:.4f}")
                    logger.info(f"[CONDITIONING] Step {i}: Other frames final mean: {other_frames_final:.4f}")
                    logger.info(f"[CONDITIONING] Step {i}: Final conditioning difference: {abs(first_frame_final - other_frames_final):.4f}")
                    
                    # Log step-specific values for comparison
                    logger.info(f"[CONDITIONING] Step {i}: step_noise_pred_final_mean: {noise_pred_final.float().mean().item():.6f}")
                    logger.info(f"[CONDITIONING] Step {i}: final_latents_mean: {latents.float().mean().item():.4f}")
                
                # Update progress
                progress_bar.update()
                if i % 5 == 0:  # Log every 5 steps
                    logger.info(f"🎬 Denoising step {i+1}/{num_inference_steps} completed")
        
        logger.info("✅ Cosmos2 denoising completed with diffuser-exact conditioning!")
        
        # Log completion timing for comparison test (matching diffusers format)
        # Note: Timing is handled at the pipeline level, not in individual stages
        
        # Update batch with final latents
        batch.latents = latents
        return batch


class Cosmos2VideoToWorldPipeline(ComposedPipelineBase):

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler"
        # Note: Removed "image_encoder" and "image_processor" as they're not needed for Cosmos
    ]
    
    def __init__(self, *args, **kwargs):
        # Set the same sigma parameters as diffusers Cosmos2VideoToWorldPipeline
        # This ensures we use the exact same sigma schedule
        self.sigma_max = 80.0
        self.sigma_min = 0.002
        self.sigma_data = 1.0
        self.final_sigmas_type = "sigma_min"
        
        super().__init__(*args, **kwargs)

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""
        
        # Configure scheduler with the same sigma parameters as diffusers
        scheduler = self.get_module("scheduler")
        if scheduler is not None:
            scheduler.register_to_config(
                sigma_max=self.sigma_max,
                sigma_min=self.sigma_min,
                sigma_data=self.sigma_data,
                final_sigmas_type=self.final_sigmas_type,
            )
            logger.info(f"🔧 Configured scheduler with sigma_max={self.sigma_max}, sigma_min={self.sigma_min}")

        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

        # Add video preprocessing stage
        # self.add_stage(stage_name="video_preprocessing_stage",
        #                stage=VideoPreprocessingStage(
        #                    vae=self.get_module("vae")
        #                ))

        self.add_stage(stage_name="conditioning_stage",
                       stage=ConditioningStage())

        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(
                           scheduler=self.get_module("scheduler")))

        # Use the specialized Cosmos latent preparation stage
        self.add_stage(stage_name="latent_preparation_stage",
                       stage=CosmosLatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer"),
                           vae=self.get_module("vae")))

        self.add_stage(stage_name="denoising_stage",
                       stage=CosmosDenoisingStage(
                           transformer=self.get_module("transformer"),
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))


EntryClass = Cosmos2VideoToWorldPipeline
