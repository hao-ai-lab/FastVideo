# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for the MAGI-2 Preview 1080p release profile."""

from __future__ import annotations

from dataclasses import dataclass, field

from fastvideo.configs.models import DiTConfig, VAEConfig
from fastvideo.configs.models.dits import (
    Magi2PreviewVideoConfig,
    Magi2RefinerVideoConfig,
)
from fastvideo.configs.models.vaes import Magi2TurboVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig


@dataclass
class Magi2PreviewPipelineConfig(PipelineConfig):
    """Configure the published 10-second MAGI-2 Preview pipeline."""

    dit_config: DiTConfig = field(default_factory=Magi2PreviewVideoConfig)
    refiner_dit_config: DiTConfig = field(default_factory=Magi2RefinerVideoConfig)
    vae_config: VAEConfig = field(default_factory=Magi2TurboVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False

    text_encoder_configs: tuple = field(default_factory=tuple)
    preprocess_text_funcs: tuple = field(default_factory=tuple)
    postprocess_text_funcs: tuple = field(default_factory=tuple)

    dit_precision: str = "bf16"
    vae_precision: str = "bf16"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=tuple)

    preview_width: int = 896
    preview_height: int = 512
    output_width: int = 1920
    output_height: int = 1088
    output_frames: int = 249
    output_fps: int = 25

    preview_video_channels: int = 48
    preview_video_length: int = 32
    preview_latent_height: int = 32
    preview_latent_width: int = 56
    audio_channels: int = 64
    audio_latent_length: int = 250
    refiner_latent_height: int = 68
    refiner_latent_width: int = 120

    preview_flow_shift: float = 7.0
    preview_video_guidance_scale: float = 5.0
    preview_audio_guidance_scale: float = 7.0
    refiner_flow_shift: float = 5.0
    refiner_video_guidance_scale: float = 2.0
    refiner_audio_guidance_scale: float = 5.0
    refiner_noise_index: int = 220


__all__ = ["Magi2PreviewPipelineConfig"]
