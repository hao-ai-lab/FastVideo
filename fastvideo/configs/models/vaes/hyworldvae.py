# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class HyWorldVAEArchConfig(VAEArchConfig):
    """Architecture config for HyWorld VAE (based on HunyuanVideo 1.5 with temporal caching)."""
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 32
    block_out_channels: tuple[int, ...] = (128, 256, 512, 1024, 1024)
    layers_per_block: int = 2
    spatial_compression_ratio: int = 16
    temporal_compression_ratio: int = 4
    downsample_match_channel: bool = True
    upsample_match_channel: bool = True
    scaling_factor: float = 1.03682

    def __post_init__(self):
        self.spatial_compression_ratio: int = 2**(len(self.block_out_channels) - 1)


@dataclass
class HyWorldVAEConfig(VAEConfig):
    """Config for HyWorld VAE with temporal caching support."""
    arch_config: VAEArchConfig = field(default_factory=HyWorldVAEArchConfig)
