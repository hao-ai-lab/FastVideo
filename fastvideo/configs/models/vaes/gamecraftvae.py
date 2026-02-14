# SPDX-License-Identifier: Apache-2.0
"""
GameCraft VAE config - matches official config.json from Hunyuan-GameCraft-1.0.
"""
from dataclasses import dataclass, field

from fastvideo.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class GameCraftVAEArchConfig(VAEArchConfig):
    """Architecture config matching official AutoencoderKLCausal3D config.json."""

    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 16
    down_block_types: tuple[str, ...] = (
        "DownEncoderBlockCausal3D",
        "DownEncoderBlockCausal3D",
        "DownEncoderBlockCausal3D",
        "DownEncoderBlockCausal3D",
    )
    up_block_types: tuple[str, ...] = (
        "UpDecoderBlockCausal3D",
        "UpDecoderBlockCausal3D",
        "UpDecoderBlockCausal3D",
        "UpDecoderBlockCausal3D",
    )
    block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    act_fn: str = "silu"
    norm_num_groups: int = 32
    scaling_factor: float = 0.476986
    spatial_compression_ratio: int = 8
    temporal_compression_ratio: int = 4
    time_compression_ratio: int = 4  # alias for DecoderCausal3D
    mid_block_add_attention: bool = True
    mid_block_causal_attn: bool = True
    sample_size: int = 256  # from config.json
    sample_tsize: int = 64  # from config.json

    def __post_init__(self):
        self.spatial_compression_ratio = 2**(len(self.block_out_channels) - 1)


@dataclass
class GameCraftVAEConfig(VAEConfig):
    """Full config for GameCraft VAE."""

    arch_config: VAEArchConfig = field(default_factory=GameCraftVAEArchConfig)
