# SPDX-License-Identifier: Apache-2.0
"""Kandinsky6 audio VAE: a thin nesting wrapper, not a new architecture.

FastVideo's existing MMAudioVAE (the mel<->latent codec) nested directly under
`vae.*`, plus a `mel_converter` waveform->mel STFT frontend, matching the
checkpoint's audio_vae layout. Only the module nesting is new; the layers are
reused unmodified.

The BigVGAN-v2 mel->waveform vocoder is a separate pipeline component
(`vocoder`, loaded by `VocoderLoader` as `BigVGANV2`; see
`Kandinsky6AudioDecodingStage`). The checkpoint's `audio_vae/*.safetensors`
also bundles a copy of the vocoder weights under `vocoder.*`;
`AudioDecoderLoader` drops those keys before the strict state-dict load, since
this class has no `vocoder` submodule.

`mel_converter` is only needed to encode real audio, which T2VA/IT2VA
generation never does: it only decodes audio latents into a waveform.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.models.audio.mmaudio_vae import MMAudioVAE


class MelConverter(nn.Module):
    """Waveform -> log-mel-spectrogram STFT frontend, matching the
    diffusers reference's `MelConverter`/`get_mel_converter("44k")`. See
    module docstring: not exercised by the current (decode-only) inference
    path."""

    def __init__(
        self,
        *,
        sampling_rate: int = 44100,
        n_fft: int = 2048,
        num_mels: int = 128,
        hop_size: int = 512,
        win_size: int = 2048,
        fmin: int = 0,
        fmax: int | None = 22050,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.register_buffer("hann_window", torch.hann_window(win_size), persistent=True)
        try:
            from librosa.filters import mel as librosa_mel_fn
            mel_basis = torch.from_numpy(
                librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)).float()
        except Exception:
            # librosa is an optional eval-only dependency here (see
            # pyproject.toml's eval-audio extra), not a core one, and this
            # buffer is only ever populated by the checkpoint's own loaded
            # weights for the (currently unused, decode-only) encode path --
            # a zero filterbank still gives the loader a correctly-shaped
            # buffer to load real weights into.
            mel_basis = torch.zeros(num_mels, n_fft // 2 + 1)
        self.register_buffer("mel_basis", mel_basis, persistent=True)

    def forward(self, waveform: torch.Tensor, center: bool = False) -> torch.Tensor:
        waveform = waveform.clamp(min=-1.0, max=1.0)
        pad = (self.n_fft - self.hop_size) // 2
        padded = F.pad(waveform.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        spec = torch.stft(
            padded,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.hann_window,
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.view_as_real(spec)
        magnitude = torch.sqrt(spec.pow(2).sum(-1) + 1e-9).float()
        mel = torch.matmul(self.mel_basis, magnitude)
        return torch.log(torch.clamp(mel, min=1e-5))


class Kandinsky6AudioVAE(nn.Module):
    """Nests the reused MMAudioVAE to match the checkpoint's exact parameter
    names. Decodes latents to a mel spectrogram only -- the separate
    `vocoder` pipeline component turns that into a waveform (see module
    docstring)."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        config = dict(config)
        config.pop("_class_name", None)
        config.pop("_diffusers_version", None)
        self.scaling_factor = float(config.get("scaling_factor", 1.0))
        # Audio-latent-frames-per-raw-audio-sample, matching the diffusers
        # reference's MMAudioVAE.downsample_factor and the pipeline config's
        # audio_downsample_factor default.
        self.downsample_factor = 1024

        mode = config.get("mode", "44k")
        need_encoder = bool(config.get("need_vae_encoder", False))

        self.mel_converter = MelConverter(sampling_rate=44100, n_fft=2048, num_mels=128, hop_size=512,
                                           win_size=2048, fmin=0, fmax=22050)
        self.vae = MMAudioVAE(mode=mode, need_encoder=need_encoder)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """latents: [B, embed_dim, A] (1D-conv channel-first) -> mel: [B, num_mels, T]."""
        return self.vae.decode(latents, unnormalize_output=True)

    def encode_audio(self, waveform: torch.Tensor):
        """waveform -> mel -> VAE posterior. Not used by generation (see the module docstring)."""
        mel = self.mel_converter(waveform)
        return self.vae.encode(mel)

    def wrapped_encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform -> mean audio latent in one call, matching the
        diffusers reference's MMAudioVAE.wrapped_encode."""
        return self.encode_audio(waveform).mean

    def remove_weight_norm(self) -> "Kandinsky6AudioVAE":
        """Called by the loader after load_state_dict: MMAudioVAE's custom
        post-load weight renormalization (needs real loaded values to
        compute from)."""
        self.vae.remove_weight_norm()
        return self


EntryClass = Kandinsky6AudioVAE
