# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class SD3Transformer2DArchConfig(DiTArchConfig):
    # Diffusers SD3Transformer2DModel config fields.
    sample_size: int = 0
    patch_size: int = 0
    num_layers: int = 0
    attention_head_dim: int = 0
    joint_attention_dim: int = 0
    caption_projection_dim: int = 0
    pooled_projection_dim: int = 0
    pos_embed_max_size: int = 0
    dual_attention_layers: list[int] = field(default_factory=list)
    qk_norm: str | None = None


@dataclass
class SD3DiTConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=SD3Transformer2DArchConfig)
    prefix: str = "sd3"

