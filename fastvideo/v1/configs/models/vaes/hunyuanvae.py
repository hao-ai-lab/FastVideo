from dataclasses import dataclass
from typing import Tuple

from fastvideo.v1.configs.models.vaes import VAEConfig, VAEArchConfig

@dataclass
class HunyuanVAEArchConfig(VAEArchConfig):
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 16
    down_block_types: Tuple[str, ...] = (
        "HunyuanVideoDownBlock3D",
        "HunyuanVideoDownBlock3D",
        "HunyuanVideoDownBlock3D",
        "HunyuanVideoDownBlock3D",
    )
    up_block_types: Tuple[str, ...] = (
        "HunyuanVideoUpBlock3D",
        "HunyuanVideoUpBlock3D",
        "HunyuanVideoUpBlock3D",
        "HunyuanVideoUpBlock3D",
    )
    block_out_channels: Tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    act_fn: str = "silu"
    norm_num_groups: int = 32
    scaling_factor: float = 0.476986
    spatial_compression_ratio: int = 8
    temporal_compression_ratio: int = 4
    mid_block_add_attention: bool = True

@dataclass
class HunyuanVAEConfig(VAEConfig):
    arch_config: VAEArchConfig = HunyuanVAEArchConfig()