# SPDX-License-Identifier: Apache-2.0
"""
GEN3C tokenizer-backed VAE adapter.

This wrapper loads the available tokenizer checkpoint (`tokenizer.pth`) and
adapts it to GEN3C's latent-time contract (T=16 for 121 output frames).

Why this exists:
- The converted GEN3C bundle includes tokenizer-style VAE weights, not a
  standard diffusers Wan VAE contract.
- GEN3C diffusion expects 8x temporal compression (121 -> 16), while the
  available tokenizer checkpoint follows a 4x temporal path.

To bridge this at inference time, we:
- keep the inner tokenizer model as-is,
- downsample encoded latent time from inner-T to target-T for DiT input,
- upsample generated latent time back to inner-T before decoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from fastvideo.logger import init_logger

logger = init_logger(__name__)


@dataclass
class _TensorLatentDist:
    """Minimal distribution-like wrapper used by pipeline stages."""

    mean: torch.Tensor

    def mode(self) -> torch.Tensor:
        return self.mean

    def sample(self, generator: Any | None = None) -> torch.Tensor:
        _ = generator
        return self.mean


class _JITGen3CTokenizerInner(nn.Module):
    """Minimal wrapper around official tokenizer JIT encoder/decoder exports."""

    def __init__(
        self,
        *,
        encoder_path: str,
        decoder_path: str,
        mean_std_path: str,
        dtype: torch.dtype,
        device: torch.device,
        latent_channels: int = 16,
        latent_chunk_duration: int = 16,
    ) -> None:
        super().__init__()
        self._dtype = dtype
        self._forced_bf16 = False
        self.encoder = torch.jit.load(encoder_path, map_location=device).eval().to(
            device=device, dtype=dtype)
        self.decoder = torch.jit.load(decoder_path, map_location=device).eval().to(
            device=device, dtype=dtype)

        latent_mean, latent_std = torch.load(mean_std_path, map_location="cpu")
        latent_mean = latent_mean.view(latent_channels, -1)[:, :latent_chunk_duration]
        latent_std = latent_std.view(latent_channels, -1)[:, :latent_chunk_duration]

        self.register_buffer(
            "_latent_mean",
            latent_mean.to(torch.float32).view(1, latent_channels,
                                               latent_chunk_duration, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_latent_std",
            latent_std.to(torch.float32).view(1, latent_channels,
                                              latent_chunk_duration, 1, 1),
            persistent=False,
        )

    def _match_stats(self, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self._latent_mean.to(device=like.device, dtype=like.dtype)
        std = self._latent_std.to(device=like.device, dtype=like.dtype)
        t = like.shape[2]
        if mean.shape[2] == t:
            return mean, std
        if t < mean.shape[2]:
            return mean[:, :, :t], std[:, :, :t]
        # fallback for non-default lengths
        mean = torch.nn.functional.interpolate(
            mean, size=(t, 1, 1), mode="trilinear", align_corners=False)
        std = torch.nn.functional.interpolate(
            std, size=(t, 1, 1), mode="trilinear", align_corners=False)
        return mean, std

    @staticmethod
    def _module_dtype_device(module: torch.nn.Module) -> tuple[torch.dtype, torch.device]:
        for param in module.parameters():
            return param.dtype, param.device
        for buf in module.buffers():
            return buf.dtype, buf.device
        raise RuntimeError("Tokenizer JIT module has no parameters/buffers to infer dtype/device.")

    def _coerce_modules_to_bf16(self) -> None:
        if self._forced_bf16:
            return
        self.encoder = self.encoder.to(dtype=torch.bfloat16)
        self.decoder = self.decoder.to(dtype=torch.bfloat16)
        self._dtype = torch.bfloat16
        self._forced_bf16 = True
        logger.warning(
            "GEN3C tokenizer JIT hit fp16/bf16 mismatch; coercing tokenizer encoder/decoder to bf16."
        )

    def encode(self, x: torch.Tensor) -> _TensorLatentDist:
        enc_dtype, enc_device = self._module_dtype_device(self.encoder)
        x_in = x.to(device=enc_device, dtype=enc_dtype)
        try:
            with torch.autocast(device_type=enc_device.type, enabled=False):
                z = self.encoder(x_in)
        except RuntimeError as e:
            err = str(e)
            mismatch_tokens = (
                "Input type (CUDABFloat16Type) and weight type (torch.cuda.HalfTensor)",
                "Input type (torch.cuda.HalfTensor) and weight type (CUDABFloat16Type)",
            )
            if any(token in err for token in mismatch_tokens):
                self._coerce_modules_to_bf16()
                enc_dtype, enc_device = self._module_dtype_device(self.encoder)
                x_in = x.to(device=enc_device, dtype=enc_dtype)
                with torch.autocast(device_type=enc_device.type, enabled=False):
                    z = self.encoder(x_in)
            else:
                raise
        if isinstance(z, tuple):
            z = z[0]
        z = z.to(dtype=x.dtype, device=x.device)
        mean, std = self._match_stats(z)
        return _TensorLatentDist((z - mean) / std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        mean, std = self._match_stats(z)
        dec_dtype, dec_device = self._module_dtype_device(self.decoder)
        z_in = (z * std + mean).to(device=dec_device, dtype=dec_dtype)
        with torch.autocast(device_type=dec_device.type, enabled=False):
            x = self.decoder(z_in)
        if isinstance(x, tuple):
            x = x[0]
        return x.to(dtype=z.dtype, device=z.device)


class AutoencoderKLGen3CTokenizer(nn.Module):
    """
    GEN3C VAE wrapper with temporal contract adaptation.

    Interface contract:
    - `encode(x)` returns normalized latents in the *target* temporal layout.
    - `decode(z)` expects normalized latents in the *target* temporal layout.
    """

    handles_latent_norm: bool = True
    handles_latent_denorm: bool = True

    def __init__(
        self,
        inner: nn.Module,
        *,
        target_temporal_compression: int = 8,
        inner_temporal_compression: int = 4,
        spatial_compression_factor: int = 8,
        pixel_chunk_duration: int = 121,
    ) -> None:
        super().__init__()
        self.inner = inner
        self.config = getattr(inner, "config", None)
        self._target_temporal_compression = int(target_temporal_compression)
        self._inner_temporal_compression = int(inner_temporal_compression)
        self._spatial_compression_factor = int(spatial_compression_factor)
        self._pixel_chunk_duration = int(pixel_chunk_duration)

    @staticmethod
    def _extract_latents(encoder_output: Any) -> torch.Tensor:
        if hasattr(encoder_output, "latent_dist"):
            dist = encoder_output.latent_dist
            if hasattr(dist, "mode"):
                return dist.mode()
            if hasattr(dist, "mean"):
                return dist.mean
            return dist.sample()
        if hasattr(encoder_output, "mode"):
            return encoder_output.mode()
        if hasattr(encoder_output, "latents"):
            return encoder_output.latents
        if hasattr(encoder_output, "sample"):
            return encoder_output.sample()
        if isinstance(encoder_output, torch.Tensor):
            return encoder_output
        raise TypeError(f"Unsupported encoder output type: {type(encoder_output)}")

    def _inner_to_target_time(self, z_inner: torch.Tensor) -> torch.Tensor:
        if z_inner.shape[2] <= 1:
            return z_inner

        # Common GEN3C case: inner=4x, target=8x => keep every other latent frame.
        if self._target_temporal_compression == 2 * self._inner_temporal_compression:
            return z_inner[:, :, 0::2, :, :].contiguous()

        # Generic fallback: keep boundary latents and sample uniformly.
        t_inner = z_inner.shape[2]
        t_target = 1 + (t_inner - 1) * self._inner_temporal_compression // self._target_temporal_compression
        idx = torch.linspace(0, t_inner - 1, t_target, device=z_inner.device)
        idx = idx.round().long()
        return z_inner.index_select(2, idx).contiguous()

    def _target_to_inner_time(self, z_target: torch.Tensor) -> torch.Tensor:
        if z_target.shape[2] <= 1:
            return z_target

        # Common GEN3C case: inner=4x, target=8x => insert midpoint frames.
        if self._target_temporal_compression == 2 * self._inner_temporal_compression:
            b, c, t, h, w = z_target.shape
            t_inner = 2 * t - 1
            out = torch.empty(
                b, c, t_inner, h, w, device=z_target.device, dtype=z_target.dtype)
            out[:, :, 0::2, :, :] = z_target
            out[:, :, 1::2, :, :] = 0.5 * (
                z_target[:, :, :-1, :, :] + z_target[:, :, 1:, :, :]
            )
            return out.contiguous()

        # Generic fallback: linear index interpolation in time.
        t_target = z_target.shape[2]
        t_inner = 1 + (t_target - 1) * self._target_temporal_compression // self._inner_temporal_compression
        idx = torch.linspace(0, t_target - 1, t_inner, device=z_target.device)
        idx0 = idx.floor().long()
        idx1 = idx.ceil().long().clamp_max(t_target - 1)
        frac = (idx - idx0).view(1, 1, -1, 1, 1)
        z0 = z_target.index_select(2, idx0)
        z1 = z_target.index_select(2, idx1)
        return (z0 * (1.0 - frac) + z1 * frac).contiguous()

    def encode(self, x: torch.Tensor) -> _TensorLatentDist:
        z_inner = self._extract_latents(self.inner.encode(x))
        z_target = self._inner_to_target_time(z_inner)
        return _TensorLatentDist(z_target)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z_inner = self._target_to_inner_time(z)
        out = self.inner.decode(z_inner)
        return out.sample if hasattr(out, "sample") else out

    def enable_tiling(self) -> None:
        if hasattr(self.inner, "enable_tiling"):
            self.inner.enable_tiling()

    def disable_tiling(self) -> None:
        if hasattr(self.inner, "disable_tiling"):
            self.inner.disable_tiling()

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        num_pixel_frames = int(num_pixel_frames)
        if num_pixel_frames <= 1:
            return 1
        return 1 + (num_pixel_frames - 1) // self._target_temporal_compression

    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        num_latent_frames = int(num_latent_frames)
        if num_latent_frames <= 1:
            return 1
        return (num_latent_frames - 1) * self._target_temporal_compression + 1

    @property
    def spatial_compression_factor(self) -> int:
        return self._spatial_compression_factor

    @property
    def temporal_compression_factor(self) -> int:
        return self._target_temporal_compression

    @property
    def temporal_compression_ratio(self) -> int:
        return self._target_temporal_compression

    @property
    def pixel_chunk_duration(self) -> int:
        return self._pixel_chunk_duration

    @property
    def latent_chunk_duration(self) -> int:
        return self.get_latent_num_frames(self._pixel_chunk_duration)

    @classmethod
    def from_tokenizer_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
        target_temporal_compression: int = 8,
        pixel_chunk_duration: int = 121,
    ) -> "AutoencoderKLGen3CTokenizer":
        from fastvideo.models.vaes.cosmos25wanvae import Cosmos25WanVAE

        inner = Cosmos25WanVAE(device=device, dtype=dtype)
        loaded = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(loaded, dict):
            for key in ("state_dict", "model", "ema", "model_state_dict"):
                if key in loaded and isinstance(loaded[key], dict):
                    loaded = loaded[key]
                    break
        missing, unexpected = inner.load_state_dict(loaded, strict=False)
        if missing:
            logger.warning(
                "GEN3C tokenizer VAE missing keys (%d). Example: %s",
                len(missing),
                missing[:5],
            )
        if unexpected:
            logger.warning(
                "GEN3C tokenizer VAE unexpected keys (%d). Example: %s",
                len(unexpected),
                unexpected[:5],
            )
        return cls(
            inner,
            target_temporal_compression=target_temporal_compression,
            inner_temporal_compression=4,
            spatial_compression_factor=8,
            pixel_chunk_duration=pixel_chunk_duration,
        )

    @classmethod
    def from_jit_tokenizer(
        cls,
        tokenizer_dir: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
        target_temporal_compression: int = 8,
        pixel_chunk_duration: int = 121,
    ) -> "AutoencoderKLGen3CTokenizer":
        encoder_path = f"{tokenizer_dir}/encoder.jit"
        decoder_path = f"{tokenizer_dir}/decoder.jit"
        mean_std_path = f"{tokenizer_dir}/mean_std.pt"
        inner = _JITGen3CTokenizerInner(
            encoder_path=encoder_path,
            decoder_path=decoder_path,
            mean_std_path=mean_std_path,
            dtype=dtype,
            device=device,
            latent_channels=16,
            latent_chunk_duration=1 + (pixel_chunk_duration - 1) //
            target_temporal_compression,
        )
        return cls(
            inner,
            target_temporal_compression=target_temporal_compression,
            inner_temporal_compression=target_temporal_compression,
            spatial_compression_factor=8,
            pixel_chunk_duration=pixel_chunk_duration,
        )
