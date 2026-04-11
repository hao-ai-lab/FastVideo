# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class FluxTransformer2DArchConfig(DiTArchConfig):

    patch_size: int = 1
    in_channels: int = 64
    out_channels: int | None = None
    num_layers: int = 19
    num_single_layers: int = 38
    attention_head_dim: int = 128
    num_attention_heads: int = 24
    joint_attention_dim: int = 4096
    pooled_projection_dim: int = 768
    guidance_embeds: bool = True
    axes_dims_rope: tuple[int, int, int] = (16, 56, 56)


@dataclass
class FluxDiTConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=FluxTransformer2DArchConfig)
    prefix: str = "flux"
