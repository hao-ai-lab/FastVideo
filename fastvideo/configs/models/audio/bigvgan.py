# SPDX-License-Identifier: Apache-2.0
"""Reusable BigVGAN-v2 vocoder configuration."""

from dataclasses import dataclass, field

from fastvideo.configs.models.base import ArchConfig, ModelConfig


@dataclass
class BigVGANV2ArchConfig(ArchConfig):
    architectures: list[str] = field(default_factory=lambda: ["BigVGANV2"])
    sample_rate: int = 44100
    num_mels: int = 128


@dataclass
class BigVGANV2Config(ModelConfig):
    arch_config: ArchConfig = field(default_factory=BigVGANV2ArchConfig)
