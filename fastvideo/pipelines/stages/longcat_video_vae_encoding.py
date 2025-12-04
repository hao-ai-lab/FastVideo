# SPDX-License-Identifier: Apache-2.0
"""
LongCat Video VAE Encoding Stage for Video Continuation (VC) generation.

This stage handles encoding multiple video frames to latent space with
LongCat-specific normalization for VC conditioning.
"""

import PIL.Image
import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.vision_utils import normalize, numpy_to_pt, pil_to_numpy, resize
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class LongCatVideoVAEEncodingStage(PipelineStage):
    """
    Encode video frames to latent space for VC conditioning.
    
    This stage:
    1. Loads video frames from path or uses provided frames
    2. Takes the last num_cond_frames from the video
    3. Preprocesses and stacks frames
    4. Encodes via VAE to latent space
    5. Applies LongCat-specific normalization
    6. Calculates num_cond_latents
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """Encode video frames to latent for VC conditioning."""

        # Get video from batch - can be path, list of PIL images, or already loaded
        video = getattr(batch, 'video_frames', None) or getattr(
            batch, 'video_path', None)
        num_cond_frames = getattr(batch, 'num_cond_frames',
                                  13)  # Default 13 for VC

        if video is None:
            raise ValueError(
                "video_frames or video_path must be provided for VC")

        # Load video if path
        if isinstance(video, str):
            from diffusers.utils import load_video
            video = load_video(video)
            logger.info("Loaded video from path: %d frames", len(video))

        # Take last num_cond_frames
        if len(video) > num_cond_frames:
            video = video[-num_cond_frames:]
            logger.info("Using last %d frames for conditioning", num_cond_frames)
        elif len(video) < num_cond_frames:
            logger.warning(
                "Video has only %d frames, less than num_cond_frames=%d",
                len(video), num_cond_frames)
            num_cond_frames = len(video)

        # Get target dimensions
        height = batch.height
        width = batch.width

        if height is None or width is None:
            raise ValueError("height and width must be set for VC")

        # Preprocess and stack frames
        processed_frames = []
        for frame in video:
            if not isinstance(frame, PIL.Image.Image):
                raise TypeError(f"Frame must be PIL.Image, got {type(frame)}")

            frame = resize(frame, height, width, resize_mode="default")
            frame = pil_to_numpy(frame)  # Returns [1, H, W, C] then converted
            frame = numpy_to_pt(frame)  # Returns [1, C, H, W]
            frame = normalize(frame)  # to [-1, 1]
            processed_frames.append(frame)

        # Stack frames: [num_frames, C, H, W] -> [1, C, T, H, W]
        video_tensor = torch.cat(processed_frames, dim=0)  # [T, C, H, W]
        video_tensor = video_tensor.permute(1, 0, 2,
                                            3).unsqueeze(0)  # [1, C, T, H, W]
        video_tensor = video_tensor.to(get_local_torch_device(),
                                       dtype=torch.float32)

        logger.info(
            "VC: Preprocessed video tensor shape: %s", video_tensor.shape)

        # Encode via VAE
        self.vae = self.vae.to(get_local_torch_device())

        # Setup VAE precision
        vae_dtype = PRECISION_TO_TYPE[
            fastvideo_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32) and not fastvideo_args.disable_autocast

        with torch.autocast(device_type="cuda",
                            dtype=vae_dtype,
                            enabled=vae_autocast_enabled):
            if fastvideo_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()

            if not vae_autocast_enabled:
                video_tensor = video_tensor.to(vae_dtype)

            with torch.no_grad():
                encoder_output = self.vae.encode(video_tensor)
                latent = self.retrieve_latents(encoder_output, batch.generator)

        # Apply LongCat-specific normalization
        latent = self.normalize_latents(latent)

        # Calculate num_cond_latents
        # Formula: 1 + (num_cond_frames - 1) // vae_temporal_scale
        vae_temporal_scale = self.vae.config.scale_factor_temporal
        num_cond_latents = 1 + (num_cond_frames - 1) // vae_temporal_scale

        # Store in batch
        batch.video_latent = latent
        batch.num_cond_frames = num_cond_frames
        batch.num_cond_latents = num_cond_latents

        logger.info(
            "VC: Encoded %d frames to latent shape %s, num_cond_latents=%d",
            num_cond_frames, latent.shape, num_cond_latents)

        # Offload VAE if needed
        if fastvideo_args.vae_cpu_offload:
            self.vae.to("cpu")

        return batch

    def retrieve_latents(self, encoder_output, generator):
        """Sample from VAE posterior."""
        if hasattr(encoder_output, 'sample'):
            return encoder_output.sample(generator)
        elif hasattr(encoder_output, 'latent_dist'):
            return encoder_output.latent_dist.sample(generator)
        elif hasattr(encoder_output, 'latents'):
            return encoder_output.latents
        else:
            raise AttributeError("Could not access latents from encoder output")

    def normalize_latents(self, latents):
        """
        Apply LongCat-specific latent normalization.
        
        Formula: (latents - mean) / std
        """
        if not hasattr(self.vae.config, 'latents_mean') or not hasattr(
                self.vae.config, 'latents_std'):
            raise ValueError(
                "VAE config must have 'latents_mean' and 'latents_std' "
                "for LongCat normalization")

        latents_mean = torch.tensor(self.vae.config.latents_mean).view(
            1, self.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)

        latents_std = torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)

        return (latents - latents_mean) / latents_std
