# SPDX-License-Identifier: Apache-2.0
"""
Decoding stage for diffusion pipelines.
"""

import torch

from fastvideo.v1.distributed import get_torch_device
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.models.vaes.common import ParallelTiledVAE
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines.stages.base import PipelineStage
from fastvideo.v1.pipelines.stages.validators import StageValidators as V
from fastvideo.v1.pipelines.stages.validators import VerificationResult
from fastvideo.v1.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class DecodingStage(PipelineStage):
    """
    Stage for decoding latent representations into pixel space.
    
    This stage handles the decoding of latent representations into the final
    output format (e.g., pixel values).
    """

    def __init__(self, vae) -> None:
        self.vae: ParallelTiledVAE = vae

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify decoding stage inputs."""
        result = VerificationResult()
        # Denoised latents for VAE decoding: [batch_size, channels, frames, height_latents, width_latents]
        result.add_check("latents", batch.latents,
                         [V.is_tensor, V.with_dims(5)])
        return result

    def verify_output(self, batch: ForwardBatch,
                      fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify decoding stage outputs."""
        result = VerificationResult()
        # Decoded video/images: [batch_size, channels, frames, height, width]
        result.add_check("output", batch.output, [V.is_tensor, V.with_dims(5)])
        return result

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Decode latent representations into pixel space.
        
        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.
            
        Returns:
            The batch with decoded outputs.
        """
        self.vae = self.vae.to(get_torch_device())

        latents = batch.latents
        # TODO(will): remove this once we add input/output validation for stages
        if latents is None:
            raise ValueError("Latents must be provided")

        # Skip decoding if output type is latent
        if fastvideo_args.output_type == "latent":
            image = latents
        else:
            # Setup VAE precision
            vae_dtype = PRECISION_TO_TYPE[
                fastvideo_args.pipeline_config.vae_precision]
            vae_autocast_enabled = (vae_dtype != torch.float32
                                    ) and not fastvideo_args.disable_autocast

            if isinstance(self.vae.scaling_factor, torch.Tensor):
                latents = latents / self.vae.scaling_factor.to(
                    latents.device, latents.dtype)
            else:
                latents = latents / self.vae.scaling_factor

            # Apply shifting if needed
            if (hasattr(self.vae, "shift_factor")
                    and self.vae.shift_factor is not None):
                if isinstance(self.vae.shift_factor, torch.Tensor):
                    latents += self.vae.shift_factor.to(latents.device,
                                                        latents.dtype)
                else:
                    latents += self.vae.shift_factor

            # Decode latents
            with torch.autocast(device_type="cuda",
                                dtype=vae_dtype,
                                enabled=vae_autocast_enabled):
                if fastvideo_args.pipeline_config.vae_tiling:
                    self.vae.enable_tiling()
                # if fastvideo_args.vae_sp:
                #     self.vae.enable_parallel()
                if not vae_autocast_enabled:
                    latents = latents.to(vae_dtype)
                image = self.vae.decode(latents)

        # Normalize image to [0, 1] range
        image = (image / 2 + 0.5).clamp(0, 1)

        # Convert to CPU float32 for compatibility
        image = image.cpu().float()

        # Update batch with decoded image
        batch.output = image

        # Offload models if needed
        if hasattr(self, 'maybe_free_model_hooks'):
            self.maybe_free_model_hooks()

        self.vae.to("cpu")
        torch.cuda.empty_cache()

        return batch
