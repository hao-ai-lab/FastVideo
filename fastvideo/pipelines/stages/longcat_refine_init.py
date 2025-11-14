# SPDX-License-Identifier: Apache-2.0
"""
LongCat refinement initialization stage.

This stage prepares the latent variables for LongCat's 480p->720p refinement by:
1. Loading the stage1 (480p) video
2. Upsampling it to 720p resolution
3. Encoding it with VAE
4. Mixing with noise according to t_thresh
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.vision_utils import load_video
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage

logger = init_logger(__name__)


class LongCatRefineInitStage(PipelineStage):
    """
    Stage for initializing LongCat refinement from a stage1 (480p) video.
    
    This replicates the logic from LongCatVideoPipeline.generate_refine():
    - Load stage1_video frames
    - Upsample spatially and temporally
    - VAE encode and normalize
    - Mix with noise according to t_thresh
    """

    def __init__(self, vae) -> None:
        super().__init__()
        self.vae = vae

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Initialize latents for refinement.
        
        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.
            
        Returns:
            The batch with initialized latents for refinement.
        """
        refine_from = batch.refine_from
        if refine_from is None:
            # Not a refinement task, skip
            return batch

        logger.info(f"Initializing LongCat refinement from: {refine_from}")
        
        # Load stage1 video
        stage1_video_path = Path(refine_from)
        if not stage1_video_path.exists():
            raise FileNotFoundError(f"Stage1 video not found: {refine_from}")
        
        # Load video frames as PIL Images
        pil_images, original_fps = load_video(str(stage1_video_path), return_fps=True)
        logger.info(f"Loaded stage1 video: {len(pil_images)} frames @ {original_fps} fps")
        
        # Store in batch for reference
        batch.stage1_video = pil_images
        
        # Get target dimensions from batch
        height = batch.height
        width = batch.width
        num_frames = len(pil_images)
        spatial_refine_only = batch.spatial_refine_only
        t_thresh = batch.t_thresh
        
        # Calculate new frame count (temporal upsampling if not spatial_refine_only)
        new_num_frames = num_frames if spatial_refine_only else 2 * num_frames
        logger.info(f"Refine mode: {'spatial only' if spatial_refine_only else 'spatial + temporal'}")
        logger.info(f"Target: {width}x{height} @ {new_num_frames} frames")
        
        # Update batch.num_frames to reflect the upsampled count
        batch.num_frames = new_num_frames
        
        # Convert PIL images to tensor [T, C, H, W]
        stage1_video_tensor = torch.stack([
            torch.from_numpy(np.array(img)).permute(2, 0, 1)  # HWC -> CHW
            for img in pil_images
        ]).float()  # [T, C, H, W]
        
        device = batch.prompt_embeds[0].device
        dtype = batch.prompt_embeds[0].dtype
        stage1_video_tensor = stage1_video_tensor.to(device=device, dtype=dtype)
        
        # Rearrange to [C, T, H, W] and add batch dimension -> [1, C, T, H, W]
        video_down = stage1_video_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]
        video_down = video_down / 255.0  # Normalize to [0, 1]
        
        # Spatial downsampling followed by upsampling (to target resolution)
        # First interpolate spatially to target H, W
        video_down_resized = F.interpolate(
            video_down, 
            size=(video_down.shape[2], height, width),  # Keep T, upsample H, W
            mode='trilinear', 
            align_corners=True
        )
        
        # Then interpolate temporally to new_num_frames
        video_up = F.interpolate(
            video_down_resized,
            size=(new_num_frames, height, width),
            mode='trilinear',
            align_corners=True
        )
        
        # Rescale to [-1, 1] for VAE
        video_up = video_up * 2.0 - 1.0
        
        logger.info(f"Upsampled video shape: {video_up.shape}")
        
        # Pad video to ensure BSA compatibility
        # BSA requires latent dimensions to be divisible by bsa_latent_granularity (4)
        # AND also divisible by sp_size after splitting (for sequence parallelism)
        bsa_latent_granularity = 4
        vae_scale_factor_temporal = 4  # VAE temporal downsampling factor
        vae_scale_factor_spatial = 8   # VAE spatial downsampling factor
        
        # Get sequence parallelism size
        sp_size = fastvideo_args.sp_size if fastvideo_args.sp_size > 0 else 1
        
        current_frames = video_up.shape[2]
        current_height = video_up.shape[3]
        current_width = video_up.shape[4]
        
        # Calculate required latent dimensions
        # For temporal: must be divisible by bsa_chunk (no SP split on T)
        num_latents_t = math.ceil(current_frames / vae_scale_factor_temporal)
        num_latents_t_padded = math.ceil(num_latents_t / bsa_latent_granularity) * bsa_latent_granularity
        target_frames = num_latents_t_padded * vae_scale_factor_temporal
        
        # For spatial: must be divisible by (bsa_chunk * sp_size) since SP splits spatially
        num_latents_h = current_height // vae_scale_factor_spatial
        spatial_granularity = bsa_latent_granularity * sp_size
        num_latents_h_padded = math.ceil(num_latents_h / spatial_granularity) * spatial_granularity
        target_height = num_latents_h_padded * vae_scale_factor_spatial
        
        num_latents_w = current_width // vae_scale_factor_spatial
        num_latents_w_padded = math.ceil(num_latents_w / spatial_granularity) * spatial_granularity
        target_width = num_latents_w_padded * vae_scale_factor_spatial
        
        # Pad if needed
        pad_t = target_frames - current_frames
        pad_h = target_height - current_height
        pad_w = target_width - current_width
        
        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            logger.info(f"Padding video for BSA (sp_size={sp_size}): T {current_frames}->{target_frames}, H {current_height}->{target_height}, W {current_width}->{target_width}")
            
            # Pad frames at the end (repeat last frame)
            if pad_t > 0:
                pad_frames = video_up[:, :, -1:].repeat(1, 1, pad_t, 1, 1)
                video_up = torch.cat([video_up, pad_frames], dim=2)
            
            # Pad height and width (use F.pad with replication)
            if pad_h > 0 or pad_w > 0:
                # F.pad format: (left, right, top, bottom, front, back)
                # We want to pad bottom and right
                video_up = F.pad(video_up, (0, pad_w, 0, pad_h, 0, 0), mode='replicate')
        
        logger.info(f"Padded video shape: {video_up.shape}")
        
        # VAE encode
        logger.info("Encoding stage1 video with VAE...")
        vae_dtype = next(self.vae.parameters()).dtype
        vae_device = next(self.vae.parameters()).device
        video_up = video_up.to(dtype=vae_dtype, device=vae_device)
        
        with torch.no_grad():
            latent_dist = self.vae.encode(video_up)
            # Extract tensor from latent distribution
            if hasattr(latent_dist, 'latent_dist'):
                # Nested distribution wrapper
                latent_up = latent_dist.latent_dist.sample()
            elif hasattr(latent_dist, 'sample'):
                # DiagonalGaussianDistribution or similar
                latent_up = latent_dist.sample()
            elif hasattr(latent_dist, 'latents'):
                # Direct latents tensor
                latent_up = latent_dist.latents
            else:
                # Assume it's already a tensor
                latent_up = latent_dist
        
        # Normalize latents using VAE config
        if hasattr(self.vae.config, 'latents_mean') and hasattr(self.vae.config, 'latents_std'):
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(
                1, self.vae.config.z_dim, 1, 1, 1
            ).to(latent_up.device, latent_up.dtype)
            latents_std = torch.tensor(self.vae.config.latents_std).view(
                1, self.vae.config.z_dim, 1, 1, 1
            ).to(latent_up.device, latent_up.dtype)
            latent_up = (latent_up - latents_mean) / latents_std
        
        logger.info(f"Encoded latent shape: {latent_up.shape}")
        
        # Mix with noise according to t_thresh
        # latent_up = (1 - t_thresh) * latent_up + t_thresh * noise
        noise = torch.randn_like(latent_up).contiguous()
        latent_up = (1 - t_thresh) * latent_up + t_thresh * noise
        
        logger.info(f"Applied t_thresh={t_thresh} noise mixing")
        
        # Store in batch
        batch.latents = latent_up.to(dtype)
        batch.raw_latent_shape = latent_up.shape
        
        logger.info("LongCat refinement initialization complete")
        
        return batch

