# SPDX-License-Identifier: Apache-2.0
import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Any

import torch

from fastvideo.configs.models.base import ArchConfig, ModelConfig
from fastvideo.utils import StoreBoolean


@dataclass
class VAEArchConfig(ArchConfig):
    scaling_factor: float | torch.Tensor = 0

    temporal_compression_ratio: int = 4
    spatial_compression_ratio: int = 8

    # Additional fields from diffusers AutoencoderKL
    act_fn: str = "silu"
    block_out_channels: list[int] = field(
        default_factory=lambda: [128, 256, 512, 512])
    down_block_types: list[str] = field(default_factory=list)
    up_block_types: list[str] = field(default_factory=list)
    force_upcast: bool = False
    in_channels: int = 3
    latent_channels: int = 16
    latents_mean: list[float] | None = None
    latents_std: list[float] | None = None
    layers_per_block: int = 2
    mid_block_add_attention: bool = True
    norm_num_groups: int = 32
    out_channels: int = 3
    sample_size: int = 1024
    shift_factor: float | None = None
    use_post_quant_conv: bool = False
    use_quant_conv: bool = False


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
    use_temporal_scaling_frames: bool = True

    def __post_init__(self):
        self.blend_num_frames = self.tile_sample_min_num_frames - self.tile_sample_stride_num_frames

    @staticmethod
    def add_cli_args(parser: Any, prefix: str = "vae-config") -> Any:
        """Add CLI arguments for VAEConfig fields"""
        parser.add_argument(
            f"--{prefix}.load-encoder",
            action=StoreBoolean,
            dest=f"{prefix.replace('-', '_')}.load_encoder",
            default=VAEConfig.load_encoder,
            help="Whether to load the VAE encoder",
        )
        parser.add_argument(
            f"--{prefix}.load-decoder",
            action=StoreBoolean,
            dest=f"{prefix.replace('-', '_')}.load_decoder",
            default=VAEConfig.load_decoder,
            help="Whether to load the VAE decoder",
        )
        parser.add_argument(
            f"--{prefix}.tile-sample-min-height",
            type=int,
            dest=f"{prefix.replace('-', '_')}.tile_sample_min_height",
            default=VAEConfig.tile_sample_min_height,
            help="Minimum height for VAE tile sampling",
        )
        parser.add_argument(
            f"--{prefix}.tile-sample-min-width",
            type=int,
            dest=f"{prefix.replace('-', '_')}.tile_sample_min_width",
            default=VAEConfig.tile_sample_min_width,
            help="Minimum width for VAE tile sampling",
        )
        parser.add_argument(
            f"--{prefix}.tile-sample-min-num-frames",
            type=int,
            dest=f"{prefix.replace('-', '_')}.tile_sample_min_num_frames",
            default=VAEConfig.tile_sample_min_num_frames,
            help="Minimum number of frames for VAE tile sampling",
        )
        parser.add_argument(
            f"--{prefix}.tile-sample-stride-height",
            type=int,
            dest=f"{prefix.replace('-', '_')}.tile_sample_stride_height",
            default=VAEConfig.tile_sample_stride_height,
            help="Stride height for VAE tile sampling",
        )
        parser.add_argument(
            f"--{prefix}.tile-sample-stride-width",
            type=int,
            dest=f"{prefix.replace('-', '_')}.tile_sample_stride_width",
            default=VAEConfig.tile_sample_stride_width,
            help="Stride width for VAE tile sampling",
        )
        parser.add_argument(
            f"--{prefix}.tile-sample-stride-num-frames",
            type=int,
            dest=f"{prefix.replace('-', '_')}.tile_sample_stride_num_frames",
            default=VAEConfig.tile_sample_stride_num_frames,
            help="Stride number of frames for VAE tile sampling",
        )
        parser.add_argument(
            f"--{prefix}.blend-num-frames",
            type=int,
            dest=f"{prefix.replace('-', '_')}.blend_num_frames",
            default=VAEConfig.blend_num_frames,
            help="Number of frames to blend for VAE tile sampling",
        )
        parser.add_argument(
            f"--{prefix}.use-tiling",
            action=StoreBoolean,
            dest=f"{prefix.replace('-', '_')}.use_tiling",
            default=VAEConfig.use_tiling,
            help="Whether to use tiling for VAE",
        )
        parser.add_argument(
            f"--{prefix}.use-temporal-tiling",
            action=StoreBoolean,
            dest=f"{prefix.replace('-', '_')}.use_temporal_tiling",
            default=VAEConfig.use_temporal_tiling,
            help="Whether to use temporal tiling for VAE",
        )
        parser.add_argument(
            f"--{prefix}.use-parallel-tiling",
            action=StoreBoolean,
            dest=f"{prefix.replace('-', '_')}.use_parallel_tiling",
            default=VAEConfig.use_parallel_tiling,
            help="Whether to use parallel tiling for VAE",
        )

        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "VAEConfig":
        kwargs = {}
        for attr in dataclasses.fields(cls):
            value = getattr(args, attr.name, None)
            if value is not None:
                kwargs[attr.name] = value
        return cls(**kwargs)
