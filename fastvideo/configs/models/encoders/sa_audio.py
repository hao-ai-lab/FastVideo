# SPDX-License-Identifier: Apache-2.0
"""Config for the Stable Audio Open 1.0 VAE used by daVinci-MagiHuman.

The reference MagiHuman pipeline uses `SAAudioFeatureExtractor`, a hand-
port of Stable Audio Open 1.0's autoencoder living in
`inference/model/sa_audio/sa_audio_module.py`. HuggingFace's diffusers
library ships the exact same architecture as
`diffusers.AutoencoderOobleck`, and Stability ships a Diffusers-layout
`vae/` subdir inside the `stabilityai/stable-audio-open-1.0` repo.

So FastVideo reuses `AutoencoderOobleck` directly (same pattern as Wan
2.2 VAE reuse), just via a thin lazy-load wrapper to mirror how
`T5GemmaEncoderModel` handles the gated-repo case for the text encoder.
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
    decoder_input_channels: int = 64  # matches DiT audio_in_channels
    audio_channels: int = 2  # stereo
    sampling_rate: int = 44100

    # Gated HF repo. The FastVideo MagiHumanPipeline.load_modules path
    # will lazy-load from here (requires HF token + accepted terms on
    # https://huggingface.co/stabilityai/stable-audio-open-1.0).
    sa_audio_model_path: str = "stabilityai/stable-audio-open-1.0"
    sa_audio_dtype: str = "float32"

    _fsdp_shard_conditions: list = field(default_factory=lambda: [_is_sa_audio_model])


@dataclass
class SAAudioVAEConfig(EncoderConfig):
    arch_config: EncoderArchConfig = field(default_factory=SAAudioVAEArchConfig)

    prefix: str = "sa_audio_vae"
