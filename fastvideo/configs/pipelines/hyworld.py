# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models import DiTConfig, EncoderConfig
from fastvideo.configs.models.dits import HyWorldConfig as HyWorldDiTConfig
from fastvideo.configs.models.encoders import SiglipVisionConfig
from fastvideo.configs.pipelines.hunyuan15 import Hunyuan15T2V480PConfig


@dataclass
class HyWorldConfig(Hunyuan15T2V480PConfig):
    """Base configuration for HyWorld pipeline architecture."""

    # HyWorldConfig-specific parameters with defaults
    dit_config: DiTConfig = field(default_factory=HyWorldDiTConfig)

    # SigLIP image encoder for I2V
    image_encoder_config: EncoderConfig = field(default_factory=SiglipVisionConfig)
    image_encoder_precision: str = "fp16"
    vae_precision: str = "fp32"

    # Text encoding
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("fp16", "fp32"))

    def __post_init__(self):
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
