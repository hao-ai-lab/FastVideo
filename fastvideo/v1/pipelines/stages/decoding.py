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
            
            # Apply proper normalization like diffusers (CRITICAL FIX!)
            if hasattr(self.vae, 'config') and hasattr(self.vae.config, 'latents_mean'):
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, self.vae.config.z_dim, 1, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std)
                    .view(1, self.vae.config.z_dim, 1, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                
                # Get sigma_data from scheduler config (default to 1.0 if not available)
                sigma_data = getattr(self.vae, 'scheduler_config', None)
                if sigma_data is None:
                    sigma_data = 1.0  # Default value
                else:
                    sigma_data = getattr(sigma_data, 'sigma_data', 1.0)
                
                # Log normalization parameters
                logger.info(f"[VAE_DECODE] latents_mean shape: {list(latents_mean.shape)}")
                logger.info(f"[VAE_DECODE] latents_std shape: {list(latents_std.shape)}")
                logger.info(f"[VAE_DECODE] sigma_data: {sigma_data}")
                
                # Apply normalization exactly like diffusers
                normalized_latents = latents * latents_std / sigma_data + latents_mean
                logger.info(f"[VAE_DECODE] After normalization stats: mean: {normalized_latents.mean().item():.4f}, std: {normalized_latents.std().item():.4f}")
                logger.info(f"[VAE_DECODE] After normalization range: [{normalized_latents.min().item():.4f}, {normalized_latents.max().item():.4f}]")
                logger.info(f"[VAE_DECODE] After normalization sum: {normalized_latents.sum().item():.4f}")
                logger.info(f"[VAE_DECODE] After normalization abs_sum: {normalized_latents.abs().sum().item():.4f}")
                logger.info(f"[VAE_DECODE] After normalization abs_max: {normalized_latents.abs().max().item():.4f}")
                logger.info(f"[VAE_DECODE] After normalization norm: {normalized_latents.norm().item():.4f}")
                
                # Use normalized latents for decoding
                latents = normalized_latents
            else:
                # Fallback to original scaling if VAE config not available
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

        # Postprocess: map from [-1, 1] to [0, 1] for display, matching diffusers
        if fastvideo_args.output_type != "latent":
            image = (image / 2 + 0.5).clamp(0, 1)

        # Update batch with decoded image
        batch.output = image

        # Offload models if needed
        if hasattr(self, 'maybe_free_model_hooks'):
            self.maybe_free_model_hooks()

        self.vae.to("cpu")

        return batch
