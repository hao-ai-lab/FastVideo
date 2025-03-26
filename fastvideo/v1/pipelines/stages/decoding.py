"""
Decoding stage for diffusion pipelines.
"""

import torch
from fastvideo.v1.pipelines.stages.base import PipelineStage
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.inference_args import InferenceArgs
from fastvideo.v1.utils import PRECISION_TO_TYPE
from fastvideo.v1.logger import init_logger

logger = init_logger(__name__)


class DecodingStage(PipelineStage):
    """
    Stage for decoding latent representations into pixel space.
    
    This stage handles the decoding of latent representations into the final
    output format (e.g., pixel values).
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(
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
            vae_autocast_enabled = (vae_dtype != torch.float32
                                    ) and not inference_args.disable_autocast

            # Apply scaling/shifting if needed
            if (hasattr(self.vae.config, "shift_factor")
                    and self.vae.config.shift_factor):
                latents = (latents / self.vae.config.scaling_factor +
                           self.vae.config.shift_factor)
            else:
                latents = latents / self.vae.config.scaling_factor

            # Decode latents
            with torch.autocast(device_type="cuda",
                                dtype=vae_dtype,
                                enabled=vae_autocast_enabled):
                if inference_args.vae_tiling:
                    self.vae.enable_tiling()
                # if inference_args.vae_sp:
                #     self.vae.enable_parallel()
                image = self.vae.decode(latents)

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
