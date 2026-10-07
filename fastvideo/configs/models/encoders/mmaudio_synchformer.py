# SPDX-License-Identifier: Apache-2.0
"""Configuration for MMAudio's Synchformer visual conditioner."""

from dataclasses import dataclass, field

from fastvideo.configs.models.encoders.base import (
    ImageEncoderArchConfig,
    ImageEncoderConfig,
)


@dataclass
class MMAudioSynchformerArchConfig(ImageEncoderArchConfig):
    architectures: list[str] = field(default_factory=lambda: ["MMAudioSynchformerVisualEncoder"])
    image_size: int = 224
    num_channels: int = 3
    segment_size: int = 16
    segment_stride: int = 8
    hidden_size: int = 768
    tokens_per_segment: int = 8


@dataclass
class MMAudioSynchformerConfig(ImageEncoderConfig):
    arch_config: ImageEncoderArchConfig = field(default_factory=MMAudioSynchformerArchConfig)
    prefix: str = "synchformer"
