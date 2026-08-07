# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 sound tokenizer (AVAE) — decode path.

The Cosmos3 ``sound_tokenizer`` is an AVAE (audio VAE). Its shipped diffusers
checkpoint is **decoder-only** (``decoder.*``) in ``AutoencoderOobleck`` naming
with SnakeBeta activations and ``weight_g``/``weight_v`` weight-norm — exactly
FastVideo's native :class:`~fastvideo.models.vaes.oobleck.OobleckDecoder`
(verified bit-exact vs the framework in ``test_cosmos3_avae_parity``). Text-to-
video+sound (t2vs) only needs DECODE: the DiT generates the sound latent and
this module decodes it to a waveform, so only the decoder is ported (the
SpectrogramConvNeXt encoder is not exported in the checkpoint).

Mirrors the framework ``AVAEModel.decode``: run the Oobleck decoder, then clamp
to [-1, 1]. The VAE bottleneck's decode is the identity (the DiT already emits
the post-bottleneck latent), so there is no bottleneck step here.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from fastvideo.logger import init_logger
from fastvideo.models.vaes.oobleck import OobleckDecoder

logger = init_logger(__name__)


@dataclass
class Cosmos3SoundVAEArchConfig:
    """Cosmos3 AVAE decoder constants (from ``sound_tokenizer/config.json``)."""

    dec_dim: int = 320  # decoder base channels
    vocoder_input_dim: int = 64  # latent channels in
    dec_c_mults: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    dec_strides: list[int] = field(default_factory=lambda: [2, 4, 5, 6, 8])
    audio_channels: int = 2  # stereo
    sampling_rate: int = 48000

    @property
    def hop_size(self) -> int:
        return int(np.prod(self.dec_strides))  # 1920


class Cosmos3SoundVAE(nn.Module):
    """Decoder-only Cosmos3 AVAE: latent ``[B, z, T]`` -> waveform ``[B, C, N]``."""

    def __init__(self, arch: Cosmos3SoundVAEArchConfig | None = None) -> None:
        super().__init__()
        self.arch = arch or Cosmos3SoundVAEArchConfig()
        self.decoder = OobleckDecoder(
            channels=self.arch.dec_dim,
            input_channels=self.arch.vocoder_input_dim,
            audio_channels=self.arch.audio_channels,
            # The framework builds decoder blocks from ``reversed(dec_strides)``
            # (deepest first), so block strides are e.g. [8,6,5,4,2].
            upsampling_ratios=list(reversed(self.arch.dec_strides)),
            channel_multiples=list(self.arch.dec_c_mults),
        )

    @property
    def sample_rate(self) -> int:
        return self.arch.sampling_rate

    @property
    def audio_channels(self) -> int:
        return self.arch.audio_channels

    @property
    def hop_size(self) -> int:
        return self.arch.hop_size

    def get_latent_num_samples(self, num_audio_samples: int) -> int:
        """Latent length for a given audio length (``AVAEInterface``: ``N // hop``)."""
        return int(num_audio_samples) // self.arch.hop_size

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode normalized latent ``[B, z, T]`` to waveform ``[B, C, N]`` in [-1, 1].

        Matches ``AVAEModel.decode``: Oobleck decoder then clamp to [-1, 1] (the
        VAE bottleneck decode is identity).
        """
        audio = self.decoder(latent)  # [B, C, N]
        return audio.clamp(-1.0, 1.0)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        torch_dtype: torch.dtype | None = None,
    ) -> "Cosmos3SoundVAE":
        """Build + load the decoder from a ``sound_tokenizer`` directory.

        Reads ``config.json`` (``dec_dim`` / ``vocoder_input_dim`` /
        ``dec_c_mults`` / ``dec_strides`` / ``sampling_rate`` / ``stereo``) and
        loads the ``decoder.*`` weights (the checkpoint is decoder-only).
        """
        from safetensors.torch import load_file

        cfg_path = os.path.join(model_path, "config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        arch = Cosmos3SoundVAEArchConfig(
            dec_dim=int(cfg["dec_dim"]),
            vocoder_input_dim=int(cfg["vocoder_input_dim"]),
            dec_c_mults=list(cfg["dec_c_mults"]),
            dec_strides=list(cfg["dec_strides"]),
            audio_channels=2 if cfg.get("stereo", True) else 1,
            sampling_rate=int(cfg.get("sampling_rate", 48000)),
        )
        model = cls(arch)

        weights_path = os.path.join(model_path, "diffusion_pytorch_model.safetensors")
        state = load_file(weights_path)
        # Decoder-only checkpoint: strip the ``decoder.`` prefix.
        dec_state = {k[len("decoder."):]: v for k, v in state.items() if k.startswith("decoder.")}
        model.decoder.load_state_dict(dec_state, strict=True)
        logger.info("Loaded Cosmos3 sound AVAE decoder (%d params) from %s",
                    sum(p.numel() for p in model.parameters()), model_path)

        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)
        model.eval()
        return model


EntryClass = Cosmos3SoundVAE
