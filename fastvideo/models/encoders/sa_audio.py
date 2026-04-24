# SPDX-License-Identifier: Apache-2.0
"""Stable Audio Open 1.0 VAE wrapper.

Thin lazy-loader around the first-class `OobleckVAE` port in
`fastvideo/models/vaes/oobleck.py`. Any FastVideo pipeline that needs
audio encoding/decoding constructs one of these, then calls
`.encode(waveform)` / `.decode(latent)` at runtime; the underlying
VAE is fetched from `stabilityai/stable-audio-open-1.0/vae/` on first
call, kept under `self._sa_audio_vae_model`, and hidden from the
parent `named_parameters()` traversal so the weight loader doesn't try
to match these against the host pipeline's repo.

Interface:

    model = SAAudioVAEModel(config)
    waveform = model.decode(audio_latent)   # (B, channels, samples)
    latent = model.encode(waveform)         # (B, decoder_input_channels, L)
    model.sampling_rate                     # int, e.g. 44100
"""
from __future__ import annotations

import os

import torch
from torch import nn

from fastvideo.configs.models.encoders import EncoderConfig


class SAAudioVAEModel(nn.Module):
    """Thin lazy-loader around FastVideo's first-class `OobleckVAE`."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        arch = config.arch_config
        self.sa_audio_model_path: str = arch.sa_audio_model_path
        self.sa_audio_dtype: str = arch.sa_audio_dtype
        self.sampling_rate: int = arch.sampling_rate
        self.audio_channels: int = arch.audio_channels
        self.decoder_input_channels: int = arch.decoder_input_channels
        self._sa_audio_vae_model = None

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        # Hide the lazy-loaded HF VAE from the parent weight loader — same
        # pattern as T5GemmaEncoderModel.
        for name, param in super().named_parameters(prefix=prefix, recurse=recurse):
            if name.startswith("_sa_audio_vae_model.") or name == "_sa_audio_vae_model":
                continue
            yield name, param

    def _build(self, device: torch.device | None = None):
        # First-class FastVideo port — no `diffusers` import at runtime.
        # Same architecture as Diffusers' AutoencoderOobleck, so Stability's
        # published `vae/config.json` + safetensors load directly.
        from fastvideo.models.vaes.oobleck import OobleckVAE
        path = self.sa_audio_model_path
        if not path:
            raise ValueError(
                "sa_audio_model_path must be set; expected "
                "`stabilityai/stable-audio-open-1.0` or a local path."
            )
        dtype = getattr(torch, self.sa_audio_dtype, torch.float32)
        # OobleckVAE.from_pretrained mirrors Diffusers' API: pass
        # subfolder="vae" for HF repo root, or point at the vae dir directly.
        subfolder: str | None = "vae"
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json")):
            subfolder = None
        model = OobleckVAE.from_pretrained(
            path, subfolder=subfolder, torch_dtype=dtype,
        )
        if device is not None:
            model = model.to(device=device)
        model.eval()
        return model

    @property
    def sa_audio_vae_model(self):
        if self._sa_audio_vae_model is None:
            self._sa_audio_vae_model = self._build()
        return self._sa_audio_vae_model

    def _move_to_input_device(self, model, ref: torch.Tensor):
        # Lazy-move the model to the input tensor's device (same trick as
        # T5GemmaEncoderModel — wrapper has no params to sniff).
        if ref is None:
            return model
        first_param = next(model.parameters(), None)
        if first_param is not None and first_param.device != ref.device:
            model = model.to(device=ref.device)
            self._sa_audio_vae_model = model
        return model

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode an audio latent (shape `[B, C_latent, L]`) into a
        waveform (shape `[B, audio_channels, samples]`).

        Upstream `SAAudioFeatureExtractor.decode(latents)` (see
        daVinci-MagiHuman/inference/model/sa_audio/sa_audio_model.py:108)
        passes `latents` directly to the VAE model. We do the same.
        """
        model = self.sa_audio_vae_model
        model = self._move_to_input_device(model, latent)
        with torch.no_grad():
            out = model.decode(latent.to(next(model.parameters()).dtype))
        # Diffusers returns a ModelOutput with `.sample`; some versions
        # also accept return_dict=False. Handle both.
        if hasattr(out, "sample"):
            return out.sample
        return out

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        model = self.sa_audio_vae_model
        model = self._move_to_input_device(model, waveform)
        with torch.no_grad():
            out = model.encode(waveform.to(next(model.parameters()).dtype))
        # Our port returns `OobleckDiagonalGaussianDistribution` directly;
        # diffusers wraps it in an Output with `.latent_dist`. Handle both.
        if hasattr(out, "latent_dist"):
            return out.latent_dist.mode()
        if hasattr(out, "mode"):
            return out.mode()
        if hasattr(out, "sample") and callable(getattr(out, "sample")):
            return out.sample()
        return out


EntryClass = SAAudioVAEModel
