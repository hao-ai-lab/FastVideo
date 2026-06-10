# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models import DiTConfig, EncoderConfig
from fastvideo.configs.models.dits.matrixgame2 import MatrixGame2WanVideoConfig
from fastvideo.configs.models.encoders import WAN2_1ControlCLIPVisionConfig
from fastvideo.configs.pipelines.wan import WanI2V480PConfig


@dataclass
class MatrixGame2BaseI2V480PConfig(WanI2V480PConfig):
    dit_config: DiTConfig = field(default_factory=MatrixGame2WanVideoConfig)
    flow_shift: float | None = 5.0


@dataclass
class MatrixGame2I2V480PConfig(WanI2V480PConfig):
    dit_config: DiTConfig = field(default_factory=MatrixGame2WanVideoConfig)
    image_encoder_config: EncoderConfig = field(default_factory=WAN2_1ControlCLIPVisionConfig)
    is_causal: bool = True
    flow_shift: float | None = 5.0
    dmd_denoising_steps: list[int] | None = field(default_factory=lambda: [1000, 666, 333])
    warp_denoising_step: bool = True
    context_noise: int = 0
    num_frames_per_block: int = 3
