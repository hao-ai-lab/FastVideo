# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 audio VAE and vocoder configuration.
"""

from dataclasses import dataclass, field

from fastvideo.configs.models.base import ArchConfig, ModelConfig


@dataclass
class LTX2AudioArchConfig(ArchConfig):
    architectures: list[str] = field(default_factory=list)


@dataclass
class LTX2AudioEncoderConfig(ModelConfig):
    arch_config: ArchConfig = field(default_factory=lambda: LTX2AudioArchConfig(
        architectures=["LTX2AudioEncoder"]))


@dataclass
class LTX2AudioDecoderConfig(ModelConfig):
    arch_config: ArchConfig = field(default_factory=lambda: LTX2AudioArchConfig(
        architectures=["LTX2AudioDecoder"]))


@dataclass
class LTX2VocoderConfig(ModelConfig):
    arch_config: ArchConfig = field(default_factory=lambda: LTX2AudioArchConfig(
        architectures=["LTX2Vocoder"]))
