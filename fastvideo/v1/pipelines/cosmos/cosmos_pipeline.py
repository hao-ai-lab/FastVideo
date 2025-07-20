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
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines.stages import (ConditioningStage, DecodingStage,
                                           DenoisingStage, InputValidationStage,
                                           LatentPreparationStage,
                                           TextEncodingStage,
                                           TimestepPreparationStage)
from fastvideo.v1.pipelines.stages.base import PipelineStage

logger = init_logger(__name__)


class CosmosLatentPreparationStage(PipelineStage):
    """
    Custom latent preparation stage for Cosmos models that does frame replacement correctly.
    
    This stage matches the official Diffusers implementation by doing frame replacement
    in the latent preparation stage, not in the transformer.
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
        Latent preparation stage that does frame replacement correctly.
        
        This matches the official Diffusers implementation by:
        1. Creating noise latents
        2. Replacing frames with conditioned latents based on conditioning mask
        3. Passing the result to the transformer
        """
        
        print(f"[COSMOS_LATENT_PREP] Stage starting...")
        
        # Get parameters
        batch_size = 1  # Assuming single batch
        num_channels_latents = 16  # VAE latent dimension
        height = batch.height if batch.height is not None else 704
        width = batch.width if batch.width is not None else 1280
        num_frames = batch.num_frames if batch.num_frames is not None else 93
        device = next(self.transformer.parameters()).device
        dtype = next(self.transformer.parameters()).dtype
        
        # Calculate latent dimensions
        vae_scale_factor_temporal = 4  # Default for AutoencoderKLWan
        vae_scale_factor_spatial = 8   # Default for AutoencoderKLWan
        num_latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1
        latent_height = height // vae_scale_factor_spatial
        latent_width = width // vae_scale_factor_spatial
        
        print(f"[COSMOS_LATENT_PREP] Target frames: {num_frames}")
        print(f"[COSMOS_LATENT_PREP] Latent frames: {num_latent_frames}")
        print(f"[COSMOS_LATENT_PREP] Latent height: {latent_height}")
        print(f"[COSMOS_LATENT_PREP] Latent width: {latent_width}")
        
        # 1. Create noise latents (like official implementation)
        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        latents = torch.randn(shape, device=device, dtype=dtype)
        
        # Get sigma_max from the sigmas tensor (first element)
        sigma_max = self.scheduler.sigmas[0].item() if hasattr(self.scheduler, 'sigmas') and len(self.scheduler.sigmas) > 0 else 80.0
        latents = latents * sigma_max
        
        print(f"[COSMOS_LATENT_PREP] Created noise latents shape: {latents.shape}")
        print(f"[COSMOS_LATENT_PREP] Noise latents mean: {latents.mean().item():.4f}")
        print(f"[COSMOS_LATENT_PREP] Noise latents std: {latents.std().item():.4f}")
        
        # 2. Do frame replacement if conditioning is available
        if hasattr(batch, 'image_latent') and batch.image_latent is not None and hasattr(batch, 'condition_video_input_mask_B_C_T_H_W') and batch.condition_video_input_mask_B_C_T_H_W is not None:
            print(f"[COSMOS_LATENT_PREP] Doing frame replacement...")
            
            # Get the conditioned latents and mask
            init_latents = batch.image_latent  # These are the conditioned latents
            cond_mask = batch.condition_video_input_mask_B_C_T_H_W  # This is the conditioning mask
            
            print(f"[COSMOS_LATENT_PREP] Init latents shape: {init_latents.shape}")
            print(f"[COSMOS_LATENT_PREP] Cond mask shape: {cond_mask.shape}")
            print(f"[COSMOS_LATENT_PREP] Cond mask sum: {cond_mask.sum().item()}")
            
            # Apply frame replacement: replace noise latents with conditioned latents based on mask
            # This matches the official implementation: latents = latents * (1 - cond_mask) + init_latents * cond_mask
            condition_mask_expanded = cond_mask.repeat(1, latents.shape[1], 1, 1, 1)
            
            # Ensure all tensors are on the same device
            condition_mask_expanded = condition_mask_expanded.to(latents.device, latents.dtype)
            init_latents = init_latents.to(latents.device, latents.dtype)
            
            latents = latents * (1 - condition_mask_expanded) + init_latents * condition_mask_expanded
            
            print(f"[COSMOS_LATENT_PREP] Applied frame replacement!")
            print(f"[COSMOS_LATENT_PREP] Latents after replacement mean: {latents.mean().item():.4f}")
            print(f"[COSMOS_LATENT_PREP] Latents after replacement std: {latents.std().item():.4f}")
        else:
            print(f"[COSMOS_LATENT_PREP] No conditioning available - using pure noise latents")
        
        # 3. Store the prepared latents in the batch
        batch.latents = latents
        
        print(f"[COSMOS_LATENT_PREP] Final latents shape: {batch.latents.shape}")
        print(f"[COSMOS_LATENT_PREP] Final latents mean: {batch.latents.mean().item():.4f}")
        print(f"[COSMOS_LATENT_PREP] Final latents std: {batch.latents.std().item():.4f}")
        
        return batch


class CosmosConditioningStage(PipelineStage):
    """
    Conditioning stage for Cosmos models that matches the official Diffusers implementation.
    
    This stage handles image encoding and conditioning mask creation, but does NOT do frame replacement.
    Frame replacement happens in the latent preparation stage, not here.
    """
    
    def __init__(self, vae, scheduler) -> None:
        super().__init__()
        self.vae = vae
        self.scheduler = scheduler
        
        # Get VAE scale factors
        # For AutoencoderKLWan, the attributes are stored as lists
        if hasattr(self.vae, 'temperal_downsample'):
            self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample)
            self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample)
        else:
            # Default scale factors for AutoencoderKLWan
            self.vae_scale_factor_temporal = 4  # Default temporal scale factor
            self.vae_scale_factor_spatial = 8   # Default spatial scale factor
    
    def preprocess_image_for_vae(self, image, height, width):
        """Preprocess image for VAE encoding."""
        import torch
        from fastvideo.v1.models.vision_utils import normalize, numpy_to_pt, pil_to_numpy, resize
        
        # Simple preprocessing approach
        resized_image = resize(image, height, width)
        
        # Handle case where resize returns a tensor
        if isinstance(resized_image, torch.Tensor):
            image_tensor = resized_image
        else:
            # Convert PIL image to tensor - handle different return types
            try:
                image_np = pil_to_numpy(resized_image)
            except:
                # Fallback for different return types
                image_np = numpy_to_pt(resized_image)
            image_tensor = numpy_to_pt(image_np)
        
        image_tensor = normalize(image_tensor)
        
        # Get VAE dtype to match
        vae_dtype = next(self.vae.parameters()).dtype
        device = next(self.vae.parameters()).device
        
        # Convert to tensor if it's not already
        if not isinstance(image_tensor, torch.Tensor):
            image_tensor = torch.tensor(image_tensor)
        
        # Ensure we have the correct shape [B, C, H, W]
        if image_tensor.dim() == 4:
            # Already batched
            pass
        elif image_tensor.dim() == 3:
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected tensor shape: {image_tensor.shape}")
        
        print(f"[COSMOS_CONDITIONING] After preprocessing shape: {image_tensor.shape}")
        
        return image_tensor.to(device, dtype=vae_dtype)
    
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Conditioning stage that matches the official Diffusers implementation.
        
        This stage only prepares image latents and conditioning masks.
        Frame replacement happens in the latent preparation stage.
        """
        
        # Debug prints to understand what's happening
        print(f"[COSMOS_CONDITIONING] Stage starting...")
        print(f"[COSMOS_CONDITIONING] batch.pil_image is None: {batch.pil_image is None}")
        print(f"[COSMOS_CONDITIONING] self.vae is None: {self.vae is None}")
        if batch.pil_image is not None:
            print(f"[COSMOS_CONDITIONING] Image type: {type(batch.pil_image)}")
            print(f"[COSMOS_CONDITIONING] Image size: {batch.pil_image.size}")
        if self.vae is not None:
            print(f"[COSMOS_CONDITIONING] VAE type: {type(self.vae)}")
            print(f"[COSMOS_CONDITIONING] VAE device: {next(self.vae.parameters()).device}")
        
        # Handle image conditioning if input image is provided
        if batch.pil_image is not None and self.vae is not None:
            print(f"[COSMOS_CONDITIONING] Processing input image for conditioning")
            print(f"[COSMOS_CONDITIONING] Image size: {batch.pil_image.size}")
            print(f"[COSMOS_CONDITIONING] Target dimensions: {batch.height}x{batch.width}")
            
            try:
                # 1. Preprocess image for VAE encoding
                image = self.preprocess_image_for_vae(
                    batch.pil_image, 
                    batch.height, 
                    batch.width
                )
                
                print(f"[COSMOS_CONDITIONING] Preprocessed image shape: {image.shape}")
                print(f"[COSMOS_CONDITIONING] Preprocessed image dtype: {image.dtype}")
                print(f"[COSMOS_CONDITIONING] Preprocessed image device: {image.device}")
                
                # 2. Add frame dimension to match VAE expectations [B, C, T, H, W]
                print(f"[COSMOS_CONDITIONING] Before adding frame dimension: {image.shape}")
                image_5d = image.unsqueeze(2)  # [1, 3, 1, H, W]
                print(f"[COSMOS_CONDITIONING] After adding frame dimension: {image_5d.shape}")
                
                # 3. Encode image to latents using VAE
                with torch.no_grad():
                    image_latents = self.vae.encode(image_5d).sample()
                
                print(f"[COSMOS_CONDITIONING] Image encoded successfully!")
                print(f"[COSMOS_CONDITIONING] Raw image latents shape: {image_latents.shape}")
                print(f"[COSMOS_CONDITIONING] Raw image latents mean: {image_latents.mean().item():.4f}")
                print(f"[COSMOS_CONDITIONING] Raw image latents std: {image_latents.std().item():.4f}")
                
                # 4. CRITICAL: Apply VAE normalization (this was missing!)
                device = image_latents.device
                dtype = image_latents.dtype
                
                # Get VAE normalization parameters
                # For AutoencoderKLWan, the attributes are stored as lists
                if hasattr(self.vae, 'latents_mean') and hasattr(self.vae, 'latents_std'):
                    # Convert lists to tensors
                    latents_mean = torch.tensor(self.vae.latents_mean).view(1, self.vae.z_dim, 1, 1, 1).to(device, dtype)
                    latents_std = torch.tensor(self.vae.latents_std).view(1, self.vae.z_dim, 1, 1, 1).to(device, dtype)
                elif hasattr(self.vae.config, 'latents_mean') and hasattr(self.vae.config, 'latents_std'):
                    latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(device, dtype)
                    latents_std = torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(device, dtype)
                else:
                    # For AutoencoderKLWan, use default normalization (no normalization)
                    print(f"[COSMOS_CONDITIONING] VAE doesn't have latents_mean/std, using default normalization")
                    latents_mean = torch.zeros(1, 16, 1, 1, 1, device=device, dtype=dtype)
                    latents_std = torch.ones(1, 16, 1, 1, 1, device=device, dtype=dtype)
                
                # Apply normalization: (latents - mean) / std * sigma_data
                # For Cosmos models, sigma_data is typically 1.0
                sigma_data = getattr(self.scheduler.config, 'sigma_data', 1.0)
                image_latents = (image_latents - latents_mean) / latents_std * sigma_data
                
                print(f"[COSMOS_CONDITIONING] After normalization mean: {image_latents.mean().item():.4f}")
                print(f"[COSMOS_CONDITIONING] After normalization std: {image_latents.std().item():.4f}")
                print(f"[COSMOS_CONDITIONING] VAE latents_mean: {latents_mean.mean().item():.4f}")
                print(f"[COSMOS_CONDITIONING] VAE latents_std: {latents_std.mean().item():.4f}")
                print(f"[COSMOS_CONDITIONING] Scheduler sigma_data: {sigma_data}")
                
                # 5. Handle video frame count and padding
                batch_size, channels, num_frames, height, width = image_latents.shape
                num_frames_target = batch.num_frames
                
                # Calculate latent frame counts
                num_latent_frames = (num_frames_target - 1) // self.vae_scale_factor_temporal + 1
                num_cond_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
                
                print(f"[COSMOS_CONDITIONING] Target frames: {num_frames_target}")
                print(f"[COSMOS_CONDITIONING] Latent frames: {num_latent_frames}")
                print(f"[COSMOS_CONDITIONING] Conditioning latent frames: {num_cond_latent_frames}")
                
                # 6. Expand image latents to match video frame count
                if num_latent_frames > 1:
                    # Repeat the single frame to match video length
                    image_latents_expanded = image_latents.repeat(1, 1, num_latent_frames, 1, 1)
                    print(f"[COSMOS_CONDITIONING] Expanded image latents shape: {image_latents_expanded.shape}")
                else:
                    image_latents_expanded = image_latents
                
                # 7. Create proper conditioning masks
                latent_height = batch.height // self.vae_scale_factor_spatial if batch.height is not None else 88
                latent_width = batch.width // self.vae_scale_factor_spatial if batch.width is not None else 160
                
                # Create conditioning indicator (which frames to condition)
                cond_indicator = torch.zeros(1, 1, num_latent_frames, 1, 1, 
                                          device=image_latents.device, dtype=image_latents.dtype)
                cond_indicator[:, :, :num_cond_latent_frames] = 1.0
                
                # Create padding masks
                padding_shape = (batch_size, 1, num_latent_frames, latent_height, latent_width)
                ones_padding = image_latents.new_ones(padding_shape)
                zeros_padding = image_latents.new_zeros(padding_shape)
                
                # Create conditioning mask
                cond_mask = cond_indicator * ones_padding + (1 - cond_indicator) * zeros_padding
                
                print(f"[COSMOS_CONDITIONING] Conditioning mask shape: {cond_mask.shape}")
                print(f"[COSMOS_CONDITIONING] Conditioning mask sum: {cond_mask.sum().item()}")
                print(f"[COSMOS_CONDITIONING] Conditioning indicator: {cond_indicator.squeeze().tolist()}")
                
                # 8. Handle classifier-free guidance masks
                uncond_indicator = None
                uncond_mask = None
                if hasattr(batch, 'guidance_scale') and batch.guidance_scale > 1.0:
                    uncond_indicator = torch.zeros(1, 1, num_latent_frames, 1, 1, 
                                                device=image_latents.device, dtype=image_latents.dtype)
                    uncond_indicator[:, :, :num_cond_latent_frames] = 1.0
                    uncond_mask = uncond_indicator * ones_padding + (1 - uncond_indicator) * zeros_padding
                    print(f"[COSMOS_CONDITIONING] Created uncond mask for classifier-free guidance")
                
                # 9. Store all components in batch using existing attributes
                # Store the conditioned latents (these will be used for frame replacement in latent preparation)
                batch.image_latent = image_latents_expanded  # Use existing attribute
                batch.condition_video_input_mask_B_C_T_H_W = cond_mask  # Use existing attribute
                
                # Store original components for frame replacement in transformer
                batch.original_image_latent = image_latents  # Single frame latents
                batch.original_pixel_image = image  # Pixel space image
                
                # Store additional conditioning info in batch context if available
                if hasattr(batch, '_conditioning_info'):
                    batch._conditioning_info = {
                        'cond_indicator': cond_indicator,
                        'cond_mask': cond_mask,
                        'uncond_indicator': uncond_indicator,
                        'uncond_mask': uncond_mask,
                        'num_cond_latent_frames': num_cond_latent_frames
                    }
                
                print(f"[COSMOS_CONDITIONING] Complete conditioning applied successfully!")
                print(f"[COSMOS_CONDITIONING] Image latents shape: {batch.image_latent.shape if batch.image_latent is not None else 'None'}")
                print(f"[COSMOS_CONDITIONING] Conditioning mask shape: {batch.condition_video_input_mask_B_C_T_H_W.shape if batch.condition_video_input_mask_B_C_T_H_W is not None else 'None'}")
                print(f"[COSMOS_CONDITIONING] Original image latents shape: {batch.original_image_latent.shape if batch.original_image_latent is not None else 'None'}")
                print(f"[COSMOS_CONDITIONING] Original pixel image shape: {batch.original_pixel_image.shape if batch.original_pixel_image is not None else 'None'}")
                
            except Exception as e:
                print(f"[COSMOS_CONDITIONING] Error during image encoding: {e}")
                print(f"[COSMOS_CONDITIONING] Error type: {type(e)}")
                import traceback
                print(f"[COSMOS_CONDITIONING] Full traceback: {traceback.format_exc()}")
                print(f"[COSMOS_CONDITIONING] Using text-to-video mode as fallback")
        else:
            print(f"[COSMOS_CONDITIONING] No input image or VAE available - using text-to-video mode")
        
        return batch


class Cosmos2VideoToWorldPipeline(ComposedPipelineBase):

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler"
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""

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
                       stage=CosmosConditioningStage(vae=self.get_module("vae"), scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(
                           scheduler=self.get_module("scheduler")))

        # Use the specialized Cosmos latent preparation stage
        self.add_stage(stage_name="latent_preparation_stage",
                       stage=CosmosLatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer")))

        self.add_stage(stage_name="denoising_stage",
                       stage=DenoisingStage(
                           transformer=self.get_module("transformer"),
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))


EntryClass = Cosmos2VideoToWorldPipeline
