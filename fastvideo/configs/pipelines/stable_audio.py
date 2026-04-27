# SPDX-License-Identifier: Apache-2.0
"""`PipelineConfig` for Stable Audio Open 1.0."""
from __future__ import annotations

from dataclasses import dataclass, field

from fastvideo.configs.models import VAEConfig
from fastvideo.configs.models.vaes import OobleckVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig


@dataclass
class StableAudioT2AConfig(PipelineConfig):
    """Stable Audio Open 1.0 pipeline config."""

    vae_config: VAEConfig = field(default_factory=OobleckVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False

    # `StableAudioMultiConditioner` owns its own T5; zero out the
    # parent's text-encoder slots so the length-equality validator passes.
    text_encoder_configs: tuple = field(default_factory=tuple)
    preprocess_text_funcs: tuple = field(default_factory=tuple)
    postprocess_text_funcs: tuple = field(default_factory=tuple)

    num_inference_steps: int = 100
    guidance_scale: float = 7.0
    audio_end_in_s: float = 10.0  # short-clip default; max is ~47.5s
    audio_start_in_s: float = 0.0
    sampling_rate: int = 44100
    audio_channels: int = 2

    precision: str = "fp32"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # A2A needs encode; load both halves for either path.
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
