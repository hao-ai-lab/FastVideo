# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class WorldEngineVAEArchConfig(VAEArchConfig):
    sample_size: tuple[int, int] = (360, 640)
    channels: int = 3
    latent_channels: int = 16
    encoder_ch_0: int = 64
    encoder_ch_max: int = 256
    encoder_blocks_per_stage: tuple[int, ...] = (1, 1, 1, 1)
    decoder_ch_0: int = 128
    decoder_ch_max: int = 1024
    decoder_blocks_per_stage: tuple[int, ...] = (1, 1, 1, 1)
    skip_logvar: bool = False
    scale_factor: float = 1.0
    shift_factor: float = 0.0

    def __post_init__(self):
        self.scaling_factor = self.scale_factor
        self.spatial_compression_ratio = 16
        self.temporal_compression_ratio = 1


@dataclass
class WorldEngineVAEConfig(VAEConfig):
    arch_config: WorldEngineVAEArchConfig = field(default_factory=WorldEngineVAEArchConfig)
    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False
