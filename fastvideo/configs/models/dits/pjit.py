# Copyright 2026 Linum Inc.
# SPDX-License-Identifier: Apache-2.0
"""Architecture fields of the released Linum Pyramid-JiT checkpoint."""

from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig
from fastvideo.platforms import AttentionBackendEnum


@dataclass
class PJiTArchConfig(DiTArchConfig):
    dim: int = 2944
    num_heads: int = 23
    num_layers: int = 22
    ffn_dim: int = 7936
    refiner_blocks: int = 2
    refiner_ffn_dim: int = 4096
    adaln_rank: int = 256
    text_dim: int = 7680
    text_len: int = 256
    in_channels: int = 3
    out_channels: int = 3
    freq_dim: int = 256
    eps: float = 1e-6
    bottleneck_dim: int = 1024
    patch: tuple[int, int, int] = (1, 32, 32)
    rope_max_positions: int = 1024
    rope_spatial_scale: float = 0.5
    text_layers: tuple[int, int, int] = (7, 15, 27)

    _supported_attention_backends: tuple[AttentionBackendEnum,
                                         ...] = (AttentionBackendEnum.TORCH_SDPA, AttentionBackendEnum.FLASH_ATTN)
    param_names_mapping: dict = field(default_factory=lambda: {r"^(.*)$": r"\1"})
    reverse_param_names_mapping: dict = field(default_factory=lambda: {r"^(.*)$": r"\1"})

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_heads <= 0 or self.dim % self.num_heads or self.head_dim < 6 or self.head_dim % 2:
            raise ValueError("PyramidJiT requires even-width attention heads with space for three RoPE axes")
        if len(self.patch) != 3 or min(self.patch) < 1 or self.freq_dim < 2 or self.freq_dim % 2:
            raise ValueError("PyramidJiT requires three positive patch dimensions and an even timestep width")
        self.hidden_size = self.dim
        self.num_attention_heads = self.num_heads
        self.num_channels_latents = self.out_channels

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads


@dataclass
class PJiTConfig(DiTConfig):
    arch_config: PJiTArchConfig = field(default_factory=PJiTArchConfig)
    prefix: str = "PyramidJiT"
