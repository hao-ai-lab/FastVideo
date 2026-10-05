# SPDX-License-Identifier: Apache-2.0
"""Architecture and checkpoint loading configuration for Qwen-Image-2.1."""

from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig
from fastvideo.platforms import AttentionBackendEnum


def _is_transformer_block(name: str, module) -> bool:
    parts = name.split(".")
    return len(parts) >= 2 and parts[-2] == "transformer_blocks" and parts[-1].isdigit()


@dataclass
class QwenImage21ArchConfig(DiTArchConfig):
    patch_size: int = 1
    in_channels: int = 64
    out_channels: int = 64
    num_layers: int = 32
    attention_head_dim: int = 128
    num_attention_heads: int = 32
    context_in_dim: int = 4096
    mlp_ratio: int = 3
    axes_dims_rope: tuple[int, int, int] = (16, 56, 56)
    eps: float = 1e-6
    causal_condition: bool = True
    cast_prompt_embeds_to_dit_dtype: bool = True

    _fsdp_shard_conditions: list = field(default_factory=lambda: [_is_transformer_block])
    _supported_attention_backends: tuple[AttentionBackendEnum, ...] = (
        AttentionBackendEnum.TORCH_SDPA, AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.SAGE_ATTN)
    param_names_mapping: dict = field(default_factory=lambda: {r"^(.*)$": r"\1"})
    reverse_param_names_mapping: dict = field(default_factory=lambda: {r"^(.*)$": r"\1"})

    def __post_init__(self) -> None:
        self.axes_dims_rope = tuple(self.axes_dims_rope)
        if len(self.axes_dims_rope) != 3 or any(dim <= 0 or dim % 2 for dim in self.axes_dims_rope):
            raise ValueError("axes_dims_rope must contain three positive even dimensions")
        if sum(self.axes_dims_rope) != self.attention_head_dim:
            raise ValueError("axes_dims_rope must sum to attention_head_dim")
        if self.num_layers <= 0 or self.num_attention_heads <= 0:
            raise ValueError("num_layers and num_attention_heads must be positive")
        super().__post_init__()
        self.out_channels = self.out_channels or self.in_channels
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels


@dataclass
class QwenImage21Config(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=QwenImage21ArchConfig)
    prefix: str = "QwenImage21"
