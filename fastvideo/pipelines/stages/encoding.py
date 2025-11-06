# SPDX-License-Identifier: Apache-2.0
"""
Encoding stage for diffusion pipelines.
"""

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.vaes.common import ParallelTiledVAE
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import V  # Import validators
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class EncodingStage(PipelineStage):
    """
    Stage for encoding pixel space representations into latent space.
    
    This stage handles the encoding of pixel-space video/images into latent
    representations for further processing in the diffusion pipeline.
    """

    def __init__(self, vae: ParallelTiledVAE) -> None:
        self.vae: ParallelTiledVAE = vae

    @torch.no_grad()
    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify encoding stage inputs."""
        result = VerificationResult()
        # Input video/images for VAE encoding: [batch_size, channels, frames, height, width]
        result.add_check("latents", batch.latents,
                         [V.is_tensor, V.with_dims(5)])
        return result

    def verify_output(self, batch: ForwardBatch,
                      fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify encoding stage outputs."""
        result = VerificationResult()
        # Encoded latents: [batch_size, channels, frames, height_latents, width_latents]
        result.add_check("latents", batch.latents,
                         [V.is_tensor, V.with_dims(5)])
        return result

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Encode pixel space representations into latent space.
        
        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.
            
        Returns:
            The batch with encoded latents.
        """
        assert batch.latents is not None and isinstance(batch.latents,
                                                        torch.Tensor)

        self.vae = self.vae.to(get_local_torch_device())

        # Setup VAE precision
        vae_dtype = PRECISION_TO_TYPE[
            fastvideo_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32) and not fastvideo_args.disable_autocast

        # Normalize input to [-1, 1] range (reverse of decoding normalization)
        latents = (batch.latents * 2.0 - 1.0).clamp(-1, 1)

        # Move to appropriate device and dtype
        latents = latents.to(get_local_torch_device())

        # Encode image to latents
        with torch.autocast(device_type="cuda",
                            dtype=vae_dtype,
                            enabled=vae_autocast_enabled):
            if fastvideo_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()
            # if fastvideo_args.vae_sp:
            #     self.vae.enable_parallel()
            if not vae_autocast_enabled:
                latents = latents.to(vae_dtype)
            latents = self.vae.encode(latents).mean

        # Update batch with encoded latents
        batch.latents = latents

        # Offload models if needed
        if hasattr(self, 'maybe_free_model_hooks'):
            self.maybe_free_model_hooks()

        if fastvideo_args.vae_cpu_offload:
            self.vae.to("cpu")

        return batch
