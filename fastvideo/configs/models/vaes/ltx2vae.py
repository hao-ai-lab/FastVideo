# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 VAE configuration.
"""

from dataclasses import dataclass, field

from fastvideo.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class LTX2VAEArchConfig(VAEArchConfig):
    # Mirrors LTX-2 safetensors metadata config under "vae"
    _class_name: str = "CausalVideoAutoencoder"
    dims: int = 3
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 128
    encoder_blocks: list = field(default_factory=list)
    decoder_blocks: list = field(default_factory=list)
    patch_size: int = 4
    norm_layer: str = "pixel_norm"
    latent_log_var: str = "uniform"
    encoder_spatial_padding_mode: str = "zeros"
    decoder_spatial_padding_mode: str = "reflect"
    causal_decoder: bool = False
    timestep_conditioning: bool = True
    use_quant_conv: bool = False
    scaling_factor: float = 1.0
    normalize_latent_channels: bool = False

    # Match FastVideo naming for compression ratios (LTX-2 default)
    temporal_compression_ratio: int = 8
    spatial_compression_ratio: int = 32


@dataclass
class LTX2VAEConfig(VAEConfig):
    arch_config: VAEArchConfig = field(default_factory=LTX2VAEArchConfig)

    # LTX-2 tiling defaults (match ltx_core.video_vae.TilingConfig.default()).
    ltx2_spatial_tile_size_in_pixels: int = 512
    ltx2_spatial_tile_overlap_in_pixels: int = 64
    ltx2_temporal_tile_size_in_frames: int = 64
    ltx2_temporal_tile_overlap_in_frames: int = 24
