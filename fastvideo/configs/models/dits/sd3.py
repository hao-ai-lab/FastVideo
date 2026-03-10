# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class SD3Transformer2DArchConfig(DiTArchConfig):
    # Diffusers SD3Transformer2DModel config fields.
    sample_size: int = 128
    patch_size: int = 2
    num_layers: int = 24
    attention_head_dim: int = 64
    joint_attention_dim: int = 4096
    caption_projection_dim: int = 1536
    pooled_projection_dim: int = 2048
    pos_embed_max_size: int = 384
    dual_attention_layers: list[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    qk_norm: str = "rms_norm"
    in_channels: int = 16
    out_channels: int = 16
    num_attention_heads: int = 24


@dataclass
class SD3DiTConfig(DiTConfig):
    arch_config: DiTArchConfig = field(
        default_factory=SD3Transformer2DArchConfig)
    prefix: str = "sd3"
