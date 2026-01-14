# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits import HyWorldConfig as HyWorldDiTConfig
from fastvideo.configs.models.encoders import SiglipVisionConfig
from fastvideo.configs.models.vaes import HyWorldVAEConfig
from fastvideo.configs.pipelines.hunyuan15 import Hunyuan15T2V480PConfig


@dataclass
class HyWorldConfig(Hunyuan15T2V480PConfig):
    """Base configuration for HyWorld pipeline architecture."""

    # HyWorldConfig-specific parameters with defaults
    # DiT - override with HyWorld-specific config
    dit_config: DiTConfig = field(default_factory=HyWorldDiTConfig)
    
    # VAE - use HyWorld VAE with temporal caching support
    vae_config: VAEConfig = field(default_factory=HyWorldVAEConfig)
    
    # SigLIP image encoder for I2V
    image_encoder_config: EncoderConfig = field(default_factory=SiglipVisionConfig)
    image_encoder_precision: str = "bf16"

    def __post_init__(self):
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
