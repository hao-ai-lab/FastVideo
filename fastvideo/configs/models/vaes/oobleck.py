# SPDX-License-Identifier: Apache-2.0
"""Config for the Stable Audio Open 1.0 "Oobleck" VAE.

Mirrors the per-channel `vae/config.json` shipped in
`stabilityai/stable-audio-open-1.0` 1:1 (see
`fastvideo/models/vaes/oobleck.py::OobleckVAE.from_pretrained`, which
constructs the VAE from these fields). Inherits the FastVideo VAEConfig
base so the standard `load_encoder` / `load_decoder` flags + tiling
knobs apply.

Naming: the VAE architecture is officially "Oobleck" (per Stability
AI's stable-audio-tools) — the surrounding model family is "Stable
Audio Open 1.0". This config is named after the architecture
(`OobleckVAEConfig`) since the same VAE is shared across Stable Audio
checkpoints; downstream pipelines reference it by its arch name, not
by a host-pipeline name.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from fastvideo.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class OobleckVAEArchConfig(VAEArchConfig):
    """Stable Audio Open 1.0 VAE architecture constants."""

    architectures: list[str] = field(default_factory=lambda: ["AutoencoderOobleck"])

    # From stabilityai/stable-audio-open-1.0/vae/config.json.
    encoder_hidden_size: int = 128
    downsampling_ratios: list[int] = field(default_factory=lambda: [2, 4, 4, 8, 8])
    channel_multiples: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    decoder_channels: int = 128
    decoder_input_channels: int = 64
    audio_channels: int = 2  # stereo
    sampling_rate: int = 44100


@dataclass
class OobleckVAEConfig(VAEConfig):
    """FastVideo VAE config wrapping the Oobleck arch.

    Audio VAEs don't use the temporal/spatial tiling defaults that the
    base VAEConfig is shaped for (those exist for video VAEs); they are
    retained but irrelevant for audio.
    """

    arch_config: VAEArchConfig = field(default_factory=OobleckVAEArchConfig)

    # Audio is 1-D, so the video-VAE tiling defaults are inert. Disable
    # them so callers don't accidentally trip on tile-stride math built
    # for spatial tensors.
    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False

    # Where the FastVideo loader / pipeline-glue wrapper should fetch
    # weights from when no local path is supplied. Gated repo — caller's
    # HF token must have accepted terms on
    # https://huggingface.co/stabilityai/stable-audio-open-1.0.
    pretrained_path: str = "stabilityai/stable-audio-open-1.0"
    pretrained_subfolder: str = "vae"
    pretrained_dtype: str = "float32"
