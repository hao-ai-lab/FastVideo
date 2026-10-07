# SPDX-License-Identifier: Apache-2.0
"""MMAudio audio VAE configuration."""

from dataclasses import dataclass, field

from fastvideo.configs.models.base import ArchConfig, ModelConfig


@dataclass
class MMAudioVAEArchConfig(ArchConfig):
    architectures: list[str] = field(default_factory=lambda: ["MMAudioVAE"])
    mode: str = "44k"
    data_dim: int = 128
    embed_dim: int = 40
    hidden_dim: int = 512
    need_encoder: bool = False


@dataclass
class MMAudioVAEConfig(ModelConfig):
    arch_config: ArchConfig = field(default_factory=MMAudioVAEArchConfig)
