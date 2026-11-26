# SPDX-License-Identifier: Apache-2.0
"""Bernini-R 1.3B uses Wan weights with source-aware token conditioning."""

from dataclasses import dataclass, field

from fastvideo.models.wan.config import WanVideoArchConfig, WanVideoConfig
from fastvideo.platforms import AttentionBackendEnum


@dataclass
class BerniniRArchConfig(WanVideoArchConfig):
    num_attention_heads: int = 12
    num_layers: int = 30
    ffn_dim: int = 8960
    _supported_attention_backends: tuple[AttentionBackendEnum,
                                         ...] = (AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TORCH_SDPA)


@dataclass
class BerniniRConfig(WanVideoConfig):
    arch_config: WanVideoArchConfig = field(default_factory=BerniniRArchConfig)
    prefix: str = "BerniniR"
