from dataclasses import dataclass

from fastvideo.v1.configs.models import ArchConfig

from fastvideo.v1.configs.models.vaes.hunyuanvae import HunyuanVAEConfig, HunyuanVAEArchConfig
from fastvideo.v1.configs.models.vaes.wanvae import WanVAEConfig, WanVAEArchConfig

__all__ = [
    "HunyuanVAEConfig", "HunyuanVAEArchConfig", 
    "WanVAEConfig", "WanVAEArchConfig"
]

@dataclass
class VAEArchConfig(ArchConfig):
    temporal_compression_ratio: int = 4
    spatial_compression_ratio: int = 8
    
@dataclass
class VAEConfig:
    arch_config: VAEArchConfig

    load_encoder: bool = True
    load_decoder: bool = True

    tile_sample_min_height: int = 256
    tile_sample_min_width: int = 256
    tile_sample_min_num_frames: int = 16
    tile_sample_stride_height: int = 192
    tile_sample_stride_width: int = 192
    tile_sample_stride_num_frames: int = 12
    blend_num_frames: int = tile_sample_min_num_frames - tile_sample_stride_num_frames

    use_tiling: bool = True
    use_temporal_tiling: bool = True
    use_parallel_tiling: bool = True