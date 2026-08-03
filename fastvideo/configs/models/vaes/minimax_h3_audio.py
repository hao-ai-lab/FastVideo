# Copyright 2025 MiniMax authors and HuggingFace Team
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

from fastvideo.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class MiniMaxH3AudioVAEArchConfig(VAEArchConfig):
    """Architecture of the MiniMax H3 waveform autoencoder."""

    _class_name: str = "AutoencoderKLMiniMaxH3Audio"
    architectures: list[str] = field(default_factory=lambda: ["AutoencoderKLMiniMaxH3Audio"])

    encoder_dim: int = 64
    encoder_rates: tuple[int, ...] | list[int] = (2, 4, 4, 5, 5)
    latent_dim: int = 2048
    latent_channels: int = 32
    num_attention_heads: int = 8

    decoder_dim: int = 1024
    decoder_rates: tuple[int, ...] | list[int] = (5, 5, 2, 2, 2, 2, 2)
    decoder_kernel_sizes: tuple[int, ...] | list[int] = (9, 9, 4, 4, 4, 4, 4)
    resblock_kernel_sizes: tuple[int, ...] | list[int] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] | list[list[int]] = (
        (1, 3, 5),
        (1, 3, 5),
        (1, 3, 5),
    )

    sampling_rate: int = 32000
    latents_mean: tuple[float, ...] | list[float] | None = None
    latents_std: tuple[float, ...] | list[float] | None = None

    scaling_factor: float = 1.0
    temporal_compression_ratio: int = 1
    spatial_compression_ratio: int = 1


@dataclass
class MiniMaxH3AudioVAEConfig(VAEConfig):
    """FastVideo loader configuration for the full H3 audio VAE."""

    arch_config: MiniMaxH3AudioVAEArchConfig = field(default_factory=MiniMaxH3AudioVAEArchConfig)

    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False
