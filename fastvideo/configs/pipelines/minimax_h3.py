# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for MiniMax H3 joint video/audio generation."""

from __future__ import annotations

from dataclasses import dataclass, field

from fastvideo.configs.models import EncoderConfig, VAEConfig
from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3Config
from fastvideo.configs.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLConfig
from fastvideo.configs.models.vaes.minimax_h3_audio import MiniMaxH3AudioVAEConfig
from fastvideo.configs.models.vaes.minimax_h3_video import MiniMaxH3VideoVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig


@dataclass
class MiniMaxH3PipelineConfig(PipelineConfig):
    """Component and precision policy shared by T2VA, FL2VA, and Ref2VA."""

    dit_config: MiniMaxH3Config = field(default_factory=MiniMaxH3Config)
    vae_config: VAEConfig = field(default_factory=MiniMaxH3VideoVAEConfig)
    audio_vae_config: VAEConfig = field(default_factory=MiniMaxH3AudioVAEConfig)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(default_factory=lambda: (MiniMaxH3Qwen3VLConfig(), ))

    flow_shift: float | None = None
    embedded_cfg_scale: float | None = None
    dit_precision: str = "bf16"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16", ))
    vae_sp: bool = False

    def check_pipeline_config(self) -> None:
        super().check_pipeline_config()
        if self.flow_shift is not None:
            raise ValueError("MiniMax-H3 uses fixed video/audio scheduler shifts; flow_shift must remain unset.")


__all__ = ["MiniMaxH3PipelineConfig"]
