# SPDX-License-Identifier: Apache-2.0
# Copied and adapted from: https://github.com/sglang-ai/sglang
from dataclasses import dataclass, field
from typing import Tuple

from fastvideo.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class Flux2VAEArchConfig(VAEArchConfig):
    """Architecture configuration for Flux2 VAE model."""
    
    # Flux2 VAE-specific architecture parameters
    in_channels: int = 3
    out_channels: int = 3
    down_block_types: Tuple[str, ...] = (
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "AttnDownEncoderBlock2D",
    )
    up_block_types: Tuple[str, ...] = (
        "AttnUpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    )
    block_out_channels: Tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    act_fn: str = "silu"
    latent_channels: int = 16
    norm_num_groups: int = 32
    sample_size: int = 512
    force_upcast: bool = False
    use_quant_conv: bool = True
    use_post_quant_conv: bool = True
    mid_block_add_attention: bool = True
    batch_norm_eps: float = 1e-5
    batch_norm_momentum: float = 0.1
    patch_size: Tuple[int, int] = (1, 1)
    
    # Spatial compression (for images, this is typically 8)
    spatial_compression_ratio: int = 8
    temporal_compression_ratio: int = 1  # Images don't have temporal dimension


@dataclass
class Flux2VAEConfig(VAEConfig):
    """Configuration for Flux2 VAE model."""
    
    arch_config: Flux2VAEArchConfig = field(default_factory=Flux2VAEArchConfig)
    
    # Flux2 is an image model, so disable temporal tiling
    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False
