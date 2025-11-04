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
    def decode(self, latents: torch.Tensor,
               fastvideo_args: FastVideoArgs) -> torch.Tensor:
        """
        Decode latent representations into pixel space using VAE.
        
        Args:
            latents: Input latent tensor with shape (batch, channels, frames, height_latents, width_latents)
            fastvideo_args: Configuration containing:
                - disable_autocast: Whether to disable automatic mixed precision (default: False)
                - pipeline_config.vae_precision: VAE computation precision ("fp32", "fp16", "bf16")
                - pipeline_config.vae_tiling: Whether to enable VAE tiling for memory efficiency
            
        Returns:
            Decoded video tensor with shape (batch, channels, frames, height, width), 
            normalized to [0, 1] range and moved to CPU as float32
        """
        self.vae = self.vae.to(get_local_torch_device())
        latents = latents.to(get_local_torch_device())

        # Setup VAE precision
        vae_dtype = PRECISION_TO_TYPE[
            fastvideo_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32) and not fastvideo_args.disable_autocast

        if hasattr(self.vae, 'scaling_factor'):
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
        return image

    @torch.no_grad()
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Decode latent representations into pixel space.
        
        This method processes the batch through the VAE decoder, converting latent
        representations to pixel-space video/images. It also optionally decodes
        trajectory latents for visualization purposes.
        
        Args:
            batch: The current batch containing:
                - latents: Tensor to decode (batch, channels, frames, height_latents, width_latents)
                - return_trajectory_decoded (optional): Flag to decode trajectory latents
                - trajectory_latents (optional): Latents at different timesteps
                - trajectory_timesteps (optional): Corresponding timesteps
            fastvideo_args: Configuration containing:
                - output_type: "latent" to skip decoding, otherwise decode to pixels
                - vae_cpu_offload: Whether to offload VAE to CPU after decoding
                - model_loaded: Track VAE loading state
                - model_paths: Path to VAE model if loading needed
            
        Returns:
            Modified batch with:
                - output: Decoded frames (batch, channels, frames, height, width) as CPU float32
                - trajectory_decoded (if requested): List of decoded frames per timestep
        """
        # load vae if not already loaded (used for memory constrained devices)
        pipeline = self.pipeline() if self.pipeline else None
        if not fastvideo_args.model_loaded["vae"]:
            loader = VAELoader()
            self.vae = loader.load(fastvideo_args.model_paths["vae"],
                                   fastvideo_args)
            if pipeline:
                pipeline.add_module("vae", self.vae)
            fastvideo_args.model_loaded["vae"] = True

        if fastvideo_args.output_type == "latent":
            frames = batch.latents
        else:
            frames = self.decode(batch.latents, fastvideo_args)

        # decode trajectory latents if needed
        if batch.return_trajectory_decoded:
            batch.trajectory_decoded = []
            assert batch.trajectory_latents is not None, "batch should have trajectory latents"
            for idx in range(batch.trajectory_latents.shape[1]):
                # batch.trajectory_latents is [batch_size, timesteps, channels, frames, height, width]
                cur_latent = batch.trajectory_latents[:, idx, :, :, :, :]
                cur_timestep = batch.trajectory_timesteps[idx]
                logger.info("decoding trajectory latent for timestep: %s",
                            cur_timestep)
                decoded_frames = self.decode(cur_latent, fastvideo_args)
                batch.trajectory_decoded.append(decoded_frames.cpu().float())

        # Convert to CPU float32 for compatibility
        frames = frames.cpu().float()

        # Update batch with decoded image
        batch.output = frames

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
