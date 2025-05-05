from dataclasses import dataclass, field
from typing import Tuple

from fastvideo.v1.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class StepVideoVAEArchConfig(VAEArchConfig):
    in_channels: int = 3
    out_channels: int = 3
    z_channels: int = 64
    num_res_blocks: int = 2
    version: int = 2
    frame_len: int = 17
    world_size: int = 1
    
    spatial_compression_ratio: int = 16
    temporal_compression_ratio: int = 8
    
    use_tiling: bool = True
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False

    tile_sample_min_height: int = 128
    tile_sample_min_width: int = 128
    tile_sample_min_num_frames: int = 17
    tile_sample_stride_height: int = 128
    tile_sample_stride_width: int = 128
    tile_sample_stride_num_frames: int = 17

    scaling_factor: float = 1.0
    # def __post_init__(self):
    #     # self.spatial_compression_ratio: int = 2**(len(self.block_out_channels) -
    #     #                                           1)


@dataclass
class StepVideoVAEConfig(VAEConfig):
    arch_config: VAEArchConfig = field(default_factory=StepVideoVAEArchConfig)
