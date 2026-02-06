# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import torch

from fastvideo.configs.models.vaes.base import VAEArchConfig, VAEConfig

@dataclass
class AutoencoderKLArchConfig(VAEArchConfig):
    _name_or_path: str = ""
    act_fn: str = "silu"
    block_out_channels: tuple[int, ...] | list[int] = field(default_factory=list)
    down_block_types: tuple[str, ...] | list[str] = field(default_factory=list)
    up_block_types: tuple[str, ...] | list[str] = field(default_factory=list)
    force_upcast: bool = True
    in_channels: int = 3
    latent_channels: int = 4
    latents_mean: tuple[float, ...] | list[float] | None = None
    latents_std: tuple[float, ...] | list[float] | None = None
    layers_per_block: int = 1
    mid_block_add_attention: bool = True
    norm_num_groups: int = 32
    out_channels: int = 3
    sample_size: int = 32
    scaling_factor: float | torch.Tensor = 0.18215
    shift_factor: float | None = None
    use_post_quant_conv: bool = True
    use_quant_conv: bool = True

    temporal_compression_ratio: int = 1
    spatial_compression_ratio: int = 8


@dataclass
class AutoencoderKLVAEConfig(VAEConfig):
    arch_config: VAEArchConfig = field(default_factory=AutoencoderKLArchConfig)

