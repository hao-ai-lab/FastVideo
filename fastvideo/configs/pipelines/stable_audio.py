# SPDX-License-Identifier: Apache-2.0
"""Stable Audio pipeline config."""
from dataclasses import dataclass, field

from fastvideo.configs.models import DiTConfig
from fastvideo.configs.models.dits.stable_audio import StableAudioDiTConfig
from fastvideo.configs.pipelines.base import PipelineConfig


@dataclass
class StableAudioPipelineConfig(PipelineConfig):
    """Config for Stable Audio text-to-audio pipeline.

    Matches stable-audio-open-1.0: 44.1kHz, Oobleck VAE, T5+seconds conditioning.
    """

    dit_config: DiTConfig = field(default_factory=StableAudioDiTConfig)

    # Audio-specific
    sample_rate: int = 44100
    sample_size: int = 2097152  # Max ~47.5s at 44.1kHz
    embedded_cfg_scale: float = 6.0
