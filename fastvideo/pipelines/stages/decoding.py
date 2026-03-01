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
from fastvideo.pipelines.stages.utils import debug_nan_check
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

    def _denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Convert normalized latents into the VAE's expected latent space."""
        # Some VAEs handle latent (de)normalization internally.
        if bool(getattr(self.vae, "handles_latent_denorm", False)):
            return latents

        cfg = getattr(self.vae, "config", None)

        # MatrixGame-style: z = z * std + mean
        if (cfg is not None and hasattr(cfg, "latents_mean")
                and hasattr(cfg, "latents_std")):
            latents_mean = torch.tensor(cfg.latents_mean,
                                        device=latents.device,
                                        dtype=latents.dtype).view(
                                            1, -1, 1, 1, 1)
            latents_std = torch.tensor(cfg.latents_std,
                                       device=latents.device,
                                       dtype=latents.dtype).view(
                                           1, -1, 1, 1, 1)
            return latents * latents_std + latents_mean

        # Diffusers-style: scaling_factor (+ optional shift_factor)
        if hasattr(self.vae, "scaling_factor"):
            sf = self.vae.scaling_factor
            if isinstance(sf, torch.Tensor):
                sf = sf.to(latents.device, latents.dtype)
                safe_sf = sf.clamp(min=1e-8)
            else:
                safe_sf = max(abs(sf), 1e-8) if sf != 0 else 1e-8
            # Avoid division by zero (e.g. default config scaling_factor=0) which produces inf
            latents = latents / safe_sf

            if hasattr(self.vae,
                       "shift_factor") and self.vae.shift_factor is not None:
                if isinstance(self.vae.shift_factor, torch.Tensor):
                    latents = latents + self.vae.shift_factor.to(
                        latents.device, latents.dtype)
                else:
                    latents = latents + self.vae.shift_factor

        # Clamp to finite range so VAE decode never receives inf (prevents NaN output)
        if not torch.isfinite(latents).all():
            latents = torch.nan_to_num(
                latents, nan=0.0, posinf=10.0, neginf=-10.0
            )
        latents = latents.clamp(-10.0, 10.0)
        return latents

    @staticmethod
    def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
        """
        Flux2: (B, 128, H', W') -> (B, 32, 2*H', 2*W').
        Inverse of 2x2 patch packing; matches diffusers Flux2Pipeline._unpatchify_latents.
        """
        batch_size, num_channels, height, width = latents.shape
        # 128 -> 32*2*2
        latents = latents.reshape(
            batch_size, num_channels // (2 * 2), 2, 2, height, width
        )
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(
            batch_size, num_channels // (2 * 2), height * 2, width * 2
        )
        return latents

    def _flux2_bn_denorm_and_unpatchify(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Flux2: BN denormalize then unpatchify packed 128ch -> 32ch for VAE.
        Uses vae.bn running_mean/running_var and vae.config.batch_norm_eps.
        Skips BN when stats are invalid (NaN, inf, or all zeros) to avoid NaNs.
        """
        running_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(
            latents.device, latents.dtype
        )
        running_var = self.vae.bn.running_var.view(1, -1, 1, 1).to(
            latents.device, latents.dtype
        )
        cfg = getattr(self.vae, "config", None)
        arch = getattr(cfg, "arch_config", None) if cfg else None
        eps = getattr(arch, "batch_norm_eps", None) or getattr(
            cfg, "batch_norm_eps", 1e-5
        )
        # Skip BN if stats are invalid (unloaded, NaN, or would produce zero std)
        mean_ok = torch.isfinite(running_mean).all().item()
        var_ok = (
            torch.isfinite(running_var).all().item()
            and (running_var + eps > 0).all().item()
        )
        if mean_ok and var_ok:
            bn_std = torch.sqrt(torch.clamp(running_var + eps, min=1e-6))
            latents = latents * bn_std + running_mean
            # Clamp to avoid inf/huge values that can cause NaN in VAE decode
            latents = latents.clamp(-10.0, 10.0)
        else:
            logger.warning(
                "Flux2 VAE BN stats invalid (NaN/inf or zero var); skipping BN denorm"
            )
        return self._unpatchify_latents(latents)

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

        # Flux2: denormalize only 32ch VAE input, not 128ch packed (scaling_factor is for VAE space)
        is_flux2_packed = (
            latents.ndim == 5
            and latents.shape[1] == 128
            and hasattr(self.vae, "bn")
            and getattr(self.vae, "post_quant_conv", None) is not None
            and self.vae.post_quant_conv.weight.shape[1] == 32
        )
        if not is_flux2_packed:
            latents = self._denormalize_latents(latents)

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
            # Image VAEs (e.g. Flux2) expect 4D (B, C, H, W); squeeze T when 5D with T=1
            squeezed_for_vae = False
            if latents.ndim == 5 and latents.shape[2] == 1:
                latents = latents.squeeze(2)
                squeezed_for_vae = True
            # Flux2 packed: 128ch packed -> BN denorm + unpatchify -> 32ch for VAE
            if (
                latents.ndim == 4
                and latents.shape[1] == 128
                and hasattr(self.vae, "bn")
                and getattr(self.vae, "post_quant_conv", None) is not None
                and self.vae.post_quant_conv.weight.shape[1] == 32
            ):
                debug_nan_check(latents, "5_before_flux2_unpatchify")
                latents = self._flux2_bn_denorm_and_unpatchify(latents)
                # Denormalize 32ch VAE input (scaling_factor / shift_factor for VAE space)
                latents = self._denormalize_latents(latents)
            debug_nan_check(latents, "5_before_vae_decode")
            image = self.vae.decode(latents)
            debug_nan_check(image, "6_after_vae_decode")
            if squeezed_for_vae:
                image = image.unsqueeze(2)

        # Normalize image to [0, 1] range
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    @torch.no_grad()
    def streaming_decode(
        self,
        latents: torch.Tensor,
        fastvideo_args: FastVideoArgs,
        cache: list[torch.Tensor | None] | None = None,
        is_first_chunk: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        """
        Decode latent representations into pixel space using VAE with streaming cache.
        
        Args:
            latents: Input latent tensor with shape (batch, channels, frames, height_latents, width_latents)
            fastvideo_args: Configuration object.
            cache: VAE cache from previous call, or None to initialize a new cache.
            is_first_chunk: Whether this is the first chunk.
            
        Returns:
            A tuple of (decoded_frames, updated_cache).
        """
        self.vae = self.vae.to(get_local_torch_device())
        latents = latents.to(get_local_torch_device())

        # Setup VAE precision
        vae_dtype = PRECISION_TO_TYPE[
            fastvideo_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32) and not fastvideo_args.disable_autocast

        latents = self._denormalize_latents(latents)

        # Initialize cache if needed
        if cache is None:
            cache = self.vae.get_streaming_cache()

        # Decode latents with streaming
        with torch.autocast(device_type="cuda",
                            dtype=vae_dtype,
                            enabled=vae_autocast_enabled):
            if fastvideo_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()
            if not vae_autocast_enabled:
                latents = latents.to(vae_dtype)
            image, cache = self.vae.streaming_decode(latents, cache,
                                                     is_first_chunk)

        # Normalize image to [0, 1] range
        image = (image / 2 + 0.5).clamp(0, 1)
        assert cache is not None, "cache should not be None after streaming_decode"
        return image, cache

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

        # Convert to float32 for compatibility
        frames = frames.to(torch.float32)

        # Crop padding if this is a LongCat refinement
        if hasattr(batch, 'num_cond_frames_added') and hasattr(
                batch, 'new_frame_size_before_padding'):
            num_cond_frames_added = batch.num_cond_frames_added
            new_frame_size = batch.new_frame_size_before_padding
            if num_cond_frames_added > 0 or frames.shape[2] != new_frame_size:
                # frames is [B, C, T, H, W], crop temporal dimension
                frames = frames[:, :,
                                num_cond_frames_added:num_cond_frames_added +
                                new_frame_size, :, :]
                logger.info(
                    "Cropped LongCat refinement padding: %s:%s, final shape: %s",
                    num_cond_frames_added,
                    num_cond_frames_added + new_frame_size, frames.shape)

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