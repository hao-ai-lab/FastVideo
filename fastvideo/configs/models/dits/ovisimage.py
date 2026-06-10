# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig


def _is_double_block(n: str, m) -> bool:
    return "transformer_blocks" in n and "single" not in n and str.isdigit(n.split(".")[-1])


def _is_single_block(n: str, m) -> bool:
    return "single_transformer_blocks" in n and str.isdigit(n.split(".")[-1])


@dataclass
class OvisImageTransformer2DModelArchConfig(DiTArchConfig):
    hidden_size: int = 3072  # num_attention_heads * attention_head_dim = 24 * 128
    num_attention_heads: int = 24
    attention_head_dim: int = 128
    num_layers: int = 6
    num_single_layers: int = 27

    in_channels: int = 64
    out_channels: int | None = None
    patch_size: int = 1

    joint_attention_dim: int = 2048
    axes_dims_rope: list[int] = field(default_factory=lambda: [16, 56, 56])
    mlp_ratio: float = 4.0

    num_channels_latents: int = 16  # in_channels=64 is this packed 16*4

    _fsdp_shard_conditions: list = field(default_factory=lambda: [_is_double_block, _is_single_block])
    _compile_conditions: list = field(default_factory=lambda: [_is_double_block, _is_single_block])


@dataclass
class OvisImageTransformer2DModelConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=OvisImageTransformer2DModelArchConfig)
    prefix: str = "OvisImage"
