# SPDX-License-Identifier: Apache-2.0
"""Lazy-loading pipeline wrapper around `OobleckVAE`.

Two reasons this exists rather than using `OobleckVAE` directly:

  1. The underlying VAE is fetched on first `encode`/`decode` call, not
     at construction — lets pipelines build the module tree on CPU
     before knowing the target device.
  2. The lazy VAE's params are hidden from `named_parameters()` so the
     FastVideo pipeline-component loader doesn't try to match Oobleck's
     safetensors against the host pipeline's converted-repo state dict.

For standalone use prefer `OobleckVAE.from_pretrained(...)` directly.
"""
from __future__ import annotations

import os

import torch
from torch import nn

from fastvideo.configs.models.vaes import OobleckVAEConfig


class SAAudioVAEModel(nn.Module):
    """Pipeline-glue lazy loader around `OobleckVAE`."""

    def __init__(self, config: OobleckVAEConfig) -> None:
        super().__init__()
        self.config = config
        arch = config.arch_config
        self.pretrained_path: str = config.pretrained_path
        self.pretrained_subfolder: str | None = config.pretrained_subfolder
        self.pretrained_dtype: str = config.pretrained_dtype
        self.sampling_rate: int = arch.sampling_rate
        self.audio_channels: int = arch.audio_channels
        self.decoder_input_channels: int = arch.decoder_input_channels
        self._oobleck_vae = None

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        # Hide the lazy-loaded VAE — its weights are fetched separately
        # and shouldn't appear in the host pipeline's loader sweep.
        for name, param in super().named_parameters(prefix=prefix, recurse=recurse):
            if name.startswith("_oobleck_vae.") or name == "_oobleck_vae":
                continue
            yield name, param

    def _build(self, device: torch.device | None = None):
        from fastvideo.models.vaes.oobleck import OobleckVAE

        path = self.pretrained_path
        if not path:
            raise ValueError(
                "OobleckVAEConfig.pretrained_path must be set; expected "
                "`stabilityai/stable-audio-open-1.0` or a local path."
            )
        dtype = getattr(torch, self.pretrained_dtype, torch.float32)
        # If the caller already pointed us at the VAE dir directly, drop
        # the subfolder. Otherwise pass through (default "vae").
        subfolder: str | None = self.pretrained_subfolder
        if subfolder and os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json")):
            subfolder = None
        model = OobleckVAE.from_pretrained(path, subfolder=subfolder, torch_dtype=dtype)
        if device is not None:
            model = model.to(device=device)
        model.eval()
        return model

    @property
    def oobleck_vae(self):
        if self._oobleck_vae is None:
            self._oobleck_vae = self._build()
        return self._oobleck_vae

    # Back-compat alias: callers that imported this earlier referred to
    # the underlying VAE as `sa_audio_vae_model`. Both names point at the
    # same object.
    @property
    def sa_audio_vae_model(self):
        return self.oobleck_vae

    def _move_to_input_device(self, model, ref: torch.Tensor):
        if ref is None:
            return model
        first_param = next(model.parameters(), None)
        if first_param is not None and first_param.device != ref.device:
            model = model.to(device=ref.device)
            self._oobleck_vae = model
        return model

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode an audio latent (`[B, C_latent, L]`) -> waveform
        (`[B, audio_channels, samples]`).
        """
        model = self.oobleck_vae
        model = self._move_to_input_device(model, latent)
        with torch.no_grad():
            out = model.decode(latent.to(next(model.parameters()).dtype))
        if hasattr(out, "sample"):
            return out.sample
        return out

    def encode(self, waveform: torch.Tensor, sample_posterior: bool = False) -> torch.Tensor:
        """Encode `[B, C_audio, samples]` -> latent `[B, C_latent, L]`.

        `sample_posterior=False` (default): deterministic mean.
        `sample_posterior=True`: stochastic sample (`mean + softplus(scale) * randn`).
        """
        model = self.oobleck_vae
        model = self._move_to_input_device(model, waveform)
        with torch.no_grad():
            out = model.encode(waveform.to(next(model.parameters()).dtype))
        if hasattr(out, "latent_dist"):
            out = out.latent_dist
        return out.sample() if sample_posterior else out.mode()


EntryClass = SAAudioVAEModel
