# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Waypoint-1-Small world model."""

from dataclasses import dataclass, field

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits.waypoint_transformer import WaypointConfig
from fastvideo.configs.models.encoders import T5Config
from fastvideo.configs.models.vaes import WorldEngineVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig


@dataclass
class WaypointT2VConfig(PipelineConfig):
    """Waypoint-1-Small interactive generation configuration."""

    dit_config: DiTConfig = field(default_factory=WaypointConfig)

    vae_config: VAEConfig = field(default_factory=WorldEngineVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False

    text_encoder_configs: tuple[EncoderConfig, ...] = field(default_factory=lambda: (T5Config(), ))

    precision: str = "bf16"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("fp32", ))

    flow_shift: float | None = None
    scheduler_sigmas: list[float] = field(default_factory=lambda: [
        1.0,
        0.8609585762023926,
        0.729332447052002,
        0.3205108940601349,
        0.0,
    ])

    is_causal: bool = True
    base_fps: int = 60

    n_buttons: int = 256
