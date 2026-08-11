# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 video VAE configuration."""

import math
from dataclasses import dataclass, field

from fastvideo.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class MiniMaxH3VideoVAEArchConfig(VAEArchConfig):
    """Architecture fields from ``AutoencoderKLMiniMaxH3``."""

    _class_name: str = "AutoencoderKLMiniMaxH3"
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 24
    block_out_channels: tuple[int, ...] = (128, 256, 256, 512, 512, 1024)
    layers_per_block: int = 2
    spatial_downsample_factors: tuple[int, ...] = (2, 2, 2, 2, 1, 1)
    temporal_downsample_factors: tuple[int, ...] = (1, 2, 2, 1, 1, 1)
    norm_num_groups: int = 32
    norm_eps: float = 1e-6
    spatial_padding_mode: str = "reflect"

    decoder_num_layers: int = 36
    decoder_num_attention_heads: int = 32
    decoder_attention_head_dim: int = 64
    decoder_num_register_tokens: int = 4
    decoder_ffn_mult: int = 4
    decoder_rope_theta: float = 100.0
    decoder_rope_dim_ratio: float = 0.75
    decoder_norm_eps: float = 1e-5

    clip_length: int = 17
    token_drop: int = 3
    latents_mean: tuple[float, ...] = (0.0, ) * 24
    latents_std: tuple[float, ...] = (1.0, ) * 24

    # MiniMax-H3 uses per-channel mean/std instead of a scalar latent scale.
    scaling_factor: float = 1.0
    temporal_compression_ratio: int = 4
    spatial_compression_ratio: int = 16

    def __post_init__(self) -> None:
        num_blocks = len(self.block_out_channels)
        if len(self.spatial_downsample_factors) != num_blocks:
            raise ValueError("`spatial_downsample_factors` must have one entry per encoder block.")
        if len(self.temporal_downsample_factors) != num_blocks:
            raise ValueError("`temporal_downsample_factors` must have one entry per encoder block.")
        if len(self.latents_mean) != self.latent_channels:
            raise ValueError("`latents_mean` must have one entry per latent channel.")
        if len(self.latents_std) != self.latent_channels:
            raise ValueError("`latents_std` must have one entry per latent channel.")

        self.spatial_compression_ratio = math.prod(self.spatial_downsample_factors)
        self.temporal_compression_ratio = math.prod(self.temporal_downsample_factors)

        rotary_dim = int(self.decoder_attention_head_dim * self.decoder_rope_dim_ratio)
        if rotary_dim % 6 != 0:
            raise ValueError("MiniMax-H3 decoder rotary dimensions must be divisible by 6.")


@dataclass
class MiniMaxH3VideoVAEConfig(VAEConfig):
    """FastVideo runtime controls for the MiniMax-H3 video VAE."""

    arch_config: MiniMaxH3VideoVAEArchConfig = field(default_factory=MiniMaxH3VideoVAEArchConfig)

    tile_sample_min_overlap_height: int = 64
    tile_sample_min_overlap_width: int = 64
