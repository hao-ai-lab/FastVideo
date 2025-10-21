# SPDX-License-Identifier: Apache-2.0
"""
Decoding stage for diffusion pipelines.
"""

import weakref

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import VAELoader
from fastvideo.models.vaes.common import ParallelTiledVAE
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class DecodingStage(PipelineStage):
    """
    Stage for decoding latent representations into pixel space.
    
    This stage handles the decoding of latent representations into the final
    output format (e.g., pixel values).
    """

    def __init__(self, vae, pipeline=None) -> None:
        self.vae: ParallelTiledVAE = vae
        self.pipeline = weakref.ref(pipeline) if pipeline else None

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

    @torch.no_grad()
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
        pipeline = self.pipeline() if self.pipeline else None
        if not fastvideo_args.model_loaded["vae"]:
            loader = VAELoader()
            self.vae = loader.load(fastvideo_args.model_paths["vae"],
                                   fastvideo_args)
            if pipeline:
                pipeline.add_module("vae", self.vae)
            fastvideo_args.model_loaded["vae"] = True

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

            # Apply latents normalization for Cosmos VAE
            # Source: /diffusers/pipelines/cosmos/pipeline_cosmos2_video2world.py:1000-1010
            if hasattr(self.vae, 'config') and hasattr(self.vae.config, 'latents_mean') and hasattr(self.vae.config, 'latents_std'):
                # Get scheduler for sigma_data
                pipeline = self.pipeline() if self.pipeline else None
                sigma_data = 1.0  # default
                if pipeline and hasattr(pipeline, 'modules') and 'scheduler' in pipeline.modules:
                    scheduler = pipeline.modules['scheduler']
                    if hasattr(scheduler, 'config') and hasattr(scheduler.config, 'sigma_data'):
                        sigma_data = scheduler.config.sigma_data

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

                latents_after_mul = latents * latents_std / sigma_data
                latents = latents_after_mul + latents_mean

            # Fallback to scaling_factor for other VAE types
            elif hasattr(self.vae, 'scaling_factor'):
                if isinstance(self.vae.scaling_factor, torch.Tensor):
                    latents = latents / self.vae.scaling_factor.to(
                        latents.device, latents.dtype)
                else:
                    latents = latents / self.vae.scaling_factor
            elif hasattr(self.vae, 'config') and hasattr(self.vae.config, 'scaling_factor'):
                latents = latents / self.vae.config.scaling_factor

            # NOTE: Skip this if we already applied latents_mean (for Cosmos VAE)
            elif (hasattr(self.vae, "shift_factor")
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
                decode_output = self.vae.decode(latents)

                # TEMPORARY: Handle diffusers VAE decode output compatibility
                if hasattr(decode_output, 'sample'):
                    # Diffusers VAE returns DecoderOutput with .sample attribute
                    image = decode_output.sample
                else:
                    # FastVideo VAE returns tensor directly
                    image = decode_output

        # Normalize image to [0, 1] range
        image = (image / 2 + 0.5).clamp(0, 1)

        # Convert to CPU float32 for compatibility
        image = image.cpu().float()

        # Update batch with decoded image
        batch.output = image

        # Offload models if needed
        if hasattr(self, 'maybe_free_model_hooks'):
            self.maybe_free_model_hooks()

        if fastvideo_args.vae_cpu_offload:
            self.vae.to("cpu")

        if torch.backends.mps.is_available():
            del self.vae
            if pipeline is not None and "vae" in pipeline.modules:
                del pipeline.modules["vae"]
            fastvideo_args.model_loaded["vae"] = False

        return batch
