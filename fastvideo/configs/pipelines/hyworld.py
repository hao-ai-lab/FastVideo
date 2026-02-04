# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models import DiTConfig, EncoderConfig
from fastvideo.configs.models.dits import HYWorldConfig as HYWorldDiTConfig
from fastvideo.configs.models.encoders import SiglipVisionConfig
from fastvideo.configs.pipelines.hunyuan15 import Hunyuan15T2V480PConfig


@dataclass
class HYWorldConfig(Hunyuan15T2V480PConfig):
    """Base configuration for HYWorld pipeline architecture."""

    # HYWorldConfig-specific parameters with defaults
    dit_config: DiTConfig = field(default_factory=HYWorldDiTConfig)

    # SigLIP image encoder for I2V
    image_encoder_config: EncoderConfig = field(
        default_factory=SiglipVisionConfig)
    image_encoder_precision: str = "fp16"
    # vae_precision: str = "fp32"

    # Text encoding
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("fp16", "fp32"))

    def __post_init__(self):
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
