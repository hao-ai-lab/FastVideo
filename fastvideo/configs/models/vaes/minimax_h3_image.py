# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3-Image VAE configuration.

The encoder is the video causal 3D CNN. The decoder is a 2D ViT, so the
rotary-width check is divisible by 4 (two spatial axes) rather than 6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from fastvideo.configs.models.vaes.base import VAEConfig
from fastvideo.configs.models.vaes.minimax_h3_video import MiniMaxH3VideoVAEArchConfig

# Published vae_diffusers/config.json statistics. Video defaults are zeros and
# must not be reused for images.
_LATENTS_MEAN = (
    0.858090341091156,
    -0.9606591463088989,
    1.0661640167236328,
    -0.5090325474739075,
    -0.2727581858634949,
    -1.3675414323806763,
    -0.2553254961967468,
    -0.26907554268836975,
    -0.5376840829849243,
    -0.0464097298681736,
    0.6657370328903198,
    0.19690127670764923,
    -0.5460608005523682,
    -0.4035342037677765,
    -0.23683024942874908,
    0.25928452610969543,
    -0.30133944749832153,
    0.211341992020607,
    -1.1206848621368408,
    0.3581933379173279,
    -0.04225143790245056,
    0.2604829967021942,
    0.22864092886447906,
    0.7056031823158264,
)
_LATENTS_STD = (
    1.2223774194717407,
    1.2767263650894165,
    1.6831774711608887,
    1.7549455165863037,
    1.5636216402053833,
    2.194143533706665,
    0.9653137922286987,
    1.0569885969161987,
    0.841948926448822,
    0.7729952931404114,
    1.8955937623977661,
    0.946841835975647,
    0.7996809482574463,
    0.44988900423049927,
    0.7197399735450745,
    0.6936293244361877,
    2.961095094680786,
    2.7694199085235596,
    3.0496184825897217,
    2.1088054180145264,
    3.276226282119751,
    3.1627357006073,
    2.2816812992095947,
    2.6127843856811523,
)


@dataclass
class MiniMaxH3ImageVAEArchConfig(MiniMaxH3VideoVAEArchConfig):
    """Image VAE: video encoder geometry plus the 24-layer 2D decoder."""

    _class_name: str = "AutoencoderKLMiniMaxH3Image"
    decoder_num_layers: int = 24
    decoder_num_attention_heads: int = 16
    decoder_rope_dim_ratio: float = 0.5
    latents_mean: tuple[float, ...] = _LATENTS_MEAN
    latents_std: tuple[float, ...] = _LATENTS_STD

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
        if rotary_dim % 4 != 0:
            raise ValueError("MiniMax-H3-Image decoder rotary dimensions must be divisible by 4.")


@dataclass
class MiniMaxH3ImageVAEConfig(VAEConfig):
    """Runtime controls for the MiniMax-H3-Image VAE. Tiling is on, matching the release."""

    arch_config: MiniMaxH3ImageVAEArchConfig = field(default_factory=MiniMaxH3ImageVAEArchConfig)
    tile_sample_min_height: int = 256
    tile_sample_min_width: int = 256
    tile_sample_min_overlap_height: int = 64
    tile_sample_min_overlap_width: int = 64
    use_tiling: bool = True
