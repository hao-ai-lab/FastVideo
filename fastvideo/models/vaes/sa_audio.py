# SPDX-License-Identifier: Apache-2.0
"""Pipeline-glue wrapper around the first-class `OobleckVAE` port.

`OobleckVAE` itself (in `fastvideo/models/vaes/oobleck.py`) is the
plain VAE — instantiate it via `OobleckVAE.from_pretrained(...)` and
call `.encode(waveform)` / `.decode(latent)` directly. Most callers
should use `OobleckVAE` directly.

This wrapper exists for two specific scenarios:

  1. **Pipelines that want lazy-load semantics** — the underlying VAE
     is fetched from `stabilityai/stable-audio-open-1.0/vae/` on the
     first `encode`/`decode` call, not at construction time. Useful
     for pipelines that build the module tree on CPU before knowing
     the target device.
  2. **Pipelines whose weight loader walks `named_parameters()`** —
     the lazy-loaded VAE's params are hidden from the parent
     traversal so FastVideo's pipeline-component loader doesn't try
     to match Oobleck's safetensors against the host pipeline's
     converted-repo state dict.

Interface:

    model = SAAudioVAEModel(config)            # nothing fetched yet
    waveform = model.decode(audio_latent)      # (B, channels, samples) — fetches on call
    latent = model.encode(waveform)            # (B, decoder_input_channels, L)
    model.sampling_rate                        # int, e.g. 44100
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
        # Hide the lazy-loaded VAE from the parent weight loader. The
        # FastVideo pipeline-component loader walks named_parameters() to
        # match against the host pipeline's converted-repo state dict;
        # Oobleck weights live in a separate HF repo (Stable Audio Open
        # 1.0) and shouldn't be expected there.
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

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode a waveform (`[B, audio_channels, samples]`) -> latent
        mean (`[B, decoder_input_channels, L]`).
        """
        model = self.oobleck_vae
        model = self._move_to_input_device(model, waveform)
        with torch.no_grad():
            out = model.encode(waveform.to(next(model.parameters()).dtype))
        if hasattr(out, "latent_dist"):
            return out.latent_dist.mode()
        if hasattr(out, "mode"):
            return out.mode()
        if hasattr(out, "sample") and callable(getattr(out, "sample")):
            return out.sample()
        return out


EntryClass = SAAudioVAEModel
