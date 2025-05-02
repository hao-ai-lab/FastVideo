from dataclasses import dataclass, field
from typing import Union

import torch

from fastvideo.v1.configs.models.base import ArchConfig, ModelConfig


@dataclass
class VAEArchConfig(ArchConfig):
    scaling_factor: Union[float, torch.tensor] = 0

    temporal_compression_ratio: int = 4
    spatial_compression_ratio: int = 8


@dataclass
class VAEConfig(ModelConfig):
    arch_config: VAEArchConfig = field(default_factory=VAEArchConfig)

    # FastVideoVAE-specific parameters
    load_encoder: bool = True
    load_decoder: bool = True

    tile_sample_min_height: int = 256
    tile_sample_min_width: int = 256
    tile_sample_min_num_frames: int = 16
    tile_sample_stride_height: int = 192
    tile_sample_stride_width: int = 192
    tile_sample_stride_num_frames: int = 12
    blend_num_frames: int = 0

    use_tiling: bool = True
    use_temporal_tiling: bool = True
    use_parallel_tiling: bool = True

    def __post_init__(self):
        self.blend_num_frames = self.tile_sample_min_num_frames - self.tile_sample_stride_num_frames
