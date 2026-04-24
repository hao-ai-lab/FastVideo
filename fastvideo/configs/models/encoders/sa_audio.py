# SPDX-License-Identifier: Apache-2.0
"""Config for the Stable Audio Open 1.0 VAE.

Wraps the Oobleck VAE from `stabilityai/stable-audio-open-1.0/vae/`
(first-class FastVideo port in `fastvideo/models/vaes/oobleck.py`).
Loaded via the thin `SAAudioVAEModel` lazy-load wrapper so any
pipeline that needs audio encoding/decoding — audio-visual diffusion,
text-to-audio, etc. — can pull it at first forward without bundling
the 620 MB VAE into each converted model repo.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from fastvideo.configs.models.encoders.base import (
    EncoderArchConfig,
    EncoderConfig,
)


def _is_sa_audio_model(n: str, m) -> bool:
    return n.endswith("sa_audio_vae_model") or n.endswith("_sa_audio_vae_model")


@dataclass
class SAAudioVAEArchConfig(EncoderArchConfig):
    architectures: list[str] = field(default_factory=lambda: ["AutoencoderOobleck"])

    # Stable Audio Open 1.0 defaults (from stabilityai/stable-audio-open-1.0/vae/config.json).
    encoder_hidden_size: int = 128
    downsampling_ratios: list[int] = field(default_factory=lambda: [2, 4, 4, 8, 8])
    channel_multiples: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    decoder_channels: int = 128
    decoder_input_channels: int = 64
    audio_channels: int = 2  # stereo
    sampling_rate: int = 44100

    # Gated HF repo — downstream pipelines lazy-load from here on first
    # forward (requires HF token + accepted terms on
    # https://huggingface.co/stabilityai/stable-audio-open-1.0).
    sa_audio_model_path: str = "stabilityai/stable-audio-open-1.0"
    sa_audio_dtype: str = "float32"

    _fsdp_shard_conditions: list = field(default_factory=lambda: [_is_sa_audio_model])


@dataclass
class SAAudioVAEConfig(EncoderConfig):
    arch_config: EncoderArchConfig = field(default_factory=SAAudioVAEArchConfig)

    prefix: str = "sa_audio_vae"
