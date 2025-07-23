# SPDX-License-Identifier: Apache-2.0
"""
Decoding stage for diffusion pipelines.
"""

import torch

from fastvideo.v1.distributed import get_local_torch_device
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
        self.vae = self.vae.to(get_local_torch_device())

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

            # Add diagnostic logging for VAE decoding
            import logging
            logger = logging.getLogger("fastvideo.diagnostics")
            logger.info(f"[VAE_DECODE] Input latents shape: {latents.shape}")
            logger.info(f"[VAE_DECODE] Input latents stats: mean: {latents.mean().item():.4f}, std: {latents.std().item():.4f}")
            logger.info(f"[VAE_DECODE] Input latents range: [{latents.min().item():.4f}, {latents.max().item():.4f}]")
            logger.info(f"[VAE_DECODE] Input latents sum: {latents.sum().item():.4f}")
            logger.info(f"[VAE_DECODE] Input latents abs_sum: {latents.abs().sum().item():.4f}")
            logger.info(f"[VAE_DECODE] Input latents abs_max: {latents.abs().max().item():.4f}")
            logger.info(f"[VAE_DECODE] Input latents norm: {latents.norm().item():.4f}")
            
            # Use simple scaling like the original pipeline (no complex normalization)
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
                    
            logger.info(f"[VAE_DECODE] After simple scaling stats: mean: {latents.mean().item():.4f}, std: {latents.std().item():.4f}")
            logger.info(f"[VAE_DECODE] After simple scaling range: [{latents.min().item():.4f}, {latents.max().item():.4f}]")

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
                
                # Decode with VAE
                image = self.vae.decode(latents)
                logger.info(f"[VAE_DECODE] VAE output shape: {list(image.shape)}")
                logger.info(f"[VAE_DECODE] VAE output stats: mean: {image.mean().item():.4f}, std: {image.std().item():.4f}")
                logger.info(f"[VAE_DECODE] VAE output range: [{image.min().item():.4f}, {image.max().item():.4f}]")
                logger.info(f"[VAE_DECODE] VAE output sum: {image.sum().item():.4f}")
                logger.info(f"[VAE_DECODE] VAE output abs_sum: {image.abs().sum().item():.4f}")
                logger.info(f"[VAE_DECODE] VAE output abs_max: {image.abs().max().item():.4f}")
                logger.info(f"[VAE_DECODE] VAE output norm: {image.norm().item():.4f}")
                
                # Log final results after normalization
                logger.info(f"[VAE_DECODE] Final video shape: {list(image.shape)}")
                logger.info(f"[VAE_DECODE] Final video stats: mean: {image.mean().item():.4f}, std: {image.std().item():.4f}")
                logger.info(f"[VAE_DECODE] Final video range: [{image.min().item():.4f}, {image.max().item():.4f}]")
                logger.info(f"[VAE_DECODE] Final video sum: {image.sum().item():.4f}")
                logger.info(f"[VAE_DECODE] Final video abs_sum: {image.abs().sum().item():.4f}")
                logger.info(f"[VAE_DECODE] Final video abs_max: {image.abs().max().item():.4f}")
                logger.info(f"[VAE_DECODE] Final video norm: {image.norm().item():.4f}")

        # Convert to CPU float32 for compatibility (no additional normalization needed)
        image = image.cpu().float()

        # Keep in [-1, 1] range like the original pipeline (no mapping to [0, 1])
        if fastvideo_args.output_type != "latent":
            image = image.clamp(-1, 1)

        # Update batch with decoded image
        batch.output = image

        # Offload models if needed
        if hasattr(self, 'maybe_free_model_hooks'):
            self.maybe_free_model_hooks()

        self.vae.to("cpu")

        # Log stage output for comparison
        try:
            from .stage_logger import log_stage_output
            log_stage_output("decoding_stage", batch, "output")
        except ImportError:
            pass  # Stage logger not available

        return batch
