# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Tuple

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class FluxArchConfig(DiTArchConfig):
    # Diffusers config fields
    attention_head_dim: int = 128
    guidance_embeds: bool = True
    in_channels: int = 64
    joint_attention_dim: int = 4096
    num_attention_heads: int = 24
    num_layers: int = 19
    num_single_layers: int = 38
    patch_size: int = 1
    pooled_projection_dim: int = 768
    qkv_bias: bool = False
    _name_or_path: str | None = None

    # FastVideo-specific defaults
    mlp_ratio: float = 4.0
    rope_axes_dim: Tuple[int, int] = (64, 64)
    rope_theta: float = 10000.0
    out_channels: int | None = None

    def __post_init__(self) -> None:
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        if sum(self.rope_axes_dim) != self.attention_head_dim:
            half = self.attention_head_dim // 2
            self.rope_axes_dim = (half, self.attention_head_dim - half)
        if self.out_channels is None:
            self.out_channels = self.in_channels
        self.num_channels_latents = self.out_channels


@dataclass
class FluxConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=FluxArchConfig)
    prefix: str = "flux"
