from dataclasses import dataclass

from fastvideo.v1.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.v1.configs.models.dits import WanVideoConfig
from fastvideo.v1.configs.models.encoders import CLIPVisionConfig, T5Config
from fastvideo.v1.configs.models.vaes import WanVAEConfig
from fastvideo.v1.configs.pipelines.base import PipelineConfig


@dataclass
class WanT2V480PConfig(PipelineConfig):
    """Base configuration for Wan T2V 1.3B pipeline architecture."""

    # WanConfig-specific parameters with defaults
    # DiT
    dit_config: DiTConfig = WanVideoConfig()
    # VAE
    vae_config: VAEConfig = WanVAEConfig()
    vae_tiling: bool = False
    vae_sp: bool = False

    # Video parameters
    use_cpu_offload: bool = True

    # Denoising stage
    flow_shift: int = 3

    # Text encoding stage
    text_encoder_config: EncoderConfig = T5Config()

    # Precision for each component
    precision: str = "bf16"
    vae_precision: str = "fp16"
    text_encoder_precision: str = "fp32"

    # WanConfig-specific added parameters

    def __post_init__(self):
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True


@dataclass
class WanI2V480PConfig(WanT2V480PConfig):
    """Base configuration for Wan I2V 14B 480P pipeline architecture."""

    # WanConfig-specific parameters with defaults

    # Precision for each component
    image_encoder_config: EncoderConfig = CLIPVisionConfig()
    image_encoder_precision: str = "fp32"

    def __post_init__(self):
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
