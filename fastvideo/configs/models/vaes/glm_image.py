# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field

from fastvideo.configs.models.vaes.autoencoder_kl import (AutoencoderKLArchConfig, AutoencoderKLVAEConfig)

_GLM_IMAGE_LATENTS_MEAN: tuple[float, ...] = (
    -0.2080078125,
    1.875,
    -0.470703125,
    -1.265625,
    -1.421875,
    0.77734375,
    -0.3671875,
    -0.9453125,
    0.318359375,
    0.7734375,
    -0.1884765625,
    -0.022216796875,
    -0.220703125,
    -1.59375,
    -0.81640625,
    -0.255859375,
)
_GLM_IMAGE_LATENTS_STD: tuple[float, ...] = (
    3.0625,
    2.203125,
    2.265625,
    4.84375,
    2.5,
    3.9375,
    2.203125,
    3.03125,
    2.1875,
    2.046875,
    2.71875,
    2.390625,
    2.390625,
    2.453125,
    2.25,
    2.15625,
)


@dataclass
class GlmImageVAEArchConfig(AutoencoderKLArchConfig):
    act_fn: str = "silu"
    block_out_channels: tuple[int, ...] = (128, 512, 1024, 1024)
    down_block_types: tuple[str, ...] = (
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    )
    up_block_types: tuple[str, ...] = (
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    )
    force_upcast: bool = True
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 16
    latents_mean: tuple[float, ...] = _GLM_IMAGE_LATENTS_MEAN
    latents_std: tuple[float, ...] = _GLM_IMAGE_LATENTS_STD
    layers_per_block: int = 3
    mid_block_add_attention: bool = False
    norm_num_groups: int = 32
    sample_size: int = 1024
    scaling_factor: float = 0.18215
    shift_factor: float | None = None
    use_quant_conv: bool = False
    use_post_quant_conv: bool = False

    temporal_compression_ratio: int = 1
    spatial_compression_ratio: int = 8


@dataclass
class GlmImageVAEConfig(AutoencoderKLVAEConfig):
    arch_config: GlmImageVAEArchConfig = field(default_factory=GlmImageVAEArchConfig)

    use_tiling: bool = True
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False

    tile_sample_min_height: int = 512
    tile_sample_min_width: int = 512
    tile_sample_stride_height: int = 384
    tile_sample_stride_width: int = 384

    load_encoder: bool = True
    load_decoder: bool = True
