"""
Decoding stage for diffusion pipelines.
"""

import torch
from typing import Optional, Union, List, Tuple
import numpy as np

from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.inference_args import InferenceArgs
from fastvideo.models.hunyuan.constants import PRECISION_TO_TYPE
from fastvideo.models.hunyuan.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from fastvideo.logger import init_logger
from diffusers.utils import BaseOutput

logger = init_logger(__name__)


class DecodingStage(PipelineStage):
    """
    Stage for decoding latent representations into pixel space.
    
    This stage handles the decoding of latent representations into the final
    output format (e.g., pixel values).
    """
    
    def _call_implementation(
        self,
        batch: ForwardBatch,
        inference_args: InferenceArgs,
    ) -> ForwardBatch:
        """
        Decode latent representations into pixel space.
        
        Args:
            batch: The current batch information.
            inference_args: The inference arguments.
            
        Returns:
            The batch with decoded outputs.
        """
        latents = batch.latents
        
        # Skip decoding if output type is latent
        if inference_args.output_type == "latent":
            image = latents
        else:
            # Setup VAE precision
            vae_dtype = PRECISION_TO_TYPE[inference_args.vae_precision]
            vae_autocast_enabled = (vae_dtype != torch.float32) and not inference_args.disable_autocast
            
            # Handle different latent shapes
            expand_temporal_dim = False
            if len(latents.shape) == 4:
                if isinstance(self.vae, AutoencoderKLCausal3D):
                    latents = latents.unsqueeze(2)
                    expand_temporal_dim = True
            elif len(latents.shape) == 5:
                pass
            else:
                raise ValueError(
                    f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}."
                )

            # Apply scaling/shifting if needed
            if (hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor):
                latents = (latents / self.vae.config.scaling_factor + self.vae.config.shift_factor)
            else:
                latents = latents / self.vae.config.scaling_factor

            # Decode latents
            with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled):
                if inference_args.vae_tiling:
                    self.vae.enable_tiling()
                image = self.vae.decode(latents, return_dict=False, generator=batch.generator)[0]

            # Handle temporal dimension if needed
            if expand_temporal_dim or image.shape[2] == 1:
                image = image.squeeze(2)

        # Normalize image to [0, 1] range
        image = (image / 2 + 0.5).clamp(0, 1)
        
        # Convert to CPU float32 for compatibility
        image = image.cpu().float()
        
        # Update batch with decoded image
        batch.videos = image
        
        # Offload models if needed
        if hasattr(self, 'maybe_free_model_hooks'):
            self.maybe_free_model_hooks()
        
        return batch 