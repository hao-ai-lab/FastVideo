# SPDX-License-Identifier: Apache-2.0
"""GLM-Image VAE configuration.

GLM-Image uses an AutoencoderKL for encoding/decoding images to/from latent space.
This is an image-only VAE (no temporal compression).
"""
from dataclasses import dataclass, field

from fastvideo.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class GlmImageVAEArchConfig(VAEArchConfig):
    """Architecture config for GLM-Image VAE (AutoencoderKL)."""

    # Standard KL-VAE parameters
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 16

    # Encoder/decoder architecture
    block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    norm_num_groups: int = 32

    # Scaling factor for latents (standard for SD-style VAEs)
    scaling_factor: float = 0.18215

    # Image-only VAE: no temporal compression
    temporal_compression_ratio: int = 1
    spatial_compression_ratio: int = 8


@dataclass
class GlmImageVAEConfig(VAEConfig):
    """Configuration for GLM-Image VAE."""

    arch_config: GlmImageVAEArchConfig = field(
        default_factory=GlmImageVAEArchConfig)

    # Tiling settings for high-resolution images
    use_tiling: bool = True
    use_temporal_tiling: bool = False  # Image model, no temporal dimension
    use_parallel_tiling: bool = False

    # Tile dimensions for memory efficiency
    tile_sample_min_height: int = 512
    tile_sample_min_width: int = 512
    tile_sample_stride_height: int = 384
    tile_sample_stride_width: int = 384

    # For image models, we need both encoder and decoder
    load_encoder: bool = True
    load_decoder: bool = True
