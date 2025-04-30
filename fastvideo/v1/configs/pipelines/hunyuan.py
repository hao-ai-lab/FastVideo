from dataclasses import dataclass

from fastvideo.v1.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.v1.configs.models.dits import HunyuanVideoConfig
from fastvideo.v1.configs.models.encoders import CLIPTextConfig, LlamaConfig
from fastvideo.v1.configs.models.vaes import HunyuanVAEConfig
from fastvideo.v1.configs.pipelines.base import PipelineConfig


@dataclass
class HunyuanConfig(PipelineConfig):
    """Base configuration for HunYuan pipeline architecture."""

    # HunyuanConfig-specific parameters with defaults
    # DiT
    dit_config: DiTConfig = HunyuanVideoConfig()
    # VAE
    vae_config: VAEConfig = HunyuanVAEConfig()
    # Denoising stage
    embedded_cfg_scale: int = 6
    flow_shift: int = 7

    # Text encoding stage
    text_encoder_config: EncoderConfig = LlamaConfig()

    # Precision for each component
    precision: str = "bf16"
    vae_precision: str = "fp16"
    text_encoder_precision: str = "fp16"

    # HunyuanConfig-specific added parameters
    # Secondary text encoder
    text_encoder_config_2: EncoderConfig = CLIPTextConfig()
    text_encoder_precision_2: str = "fp16"

    def __post_init__(self):
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True


@dataclass
class FastHunyuanConfig(HunyuanConfig):
    """Configuration specifically optimized for FastHunyuan weights."""

    # Override HunyuanConfig defaults
    flow_shift: int = 17

    # No need to re-specify guidance_scale or embedded_cfg_scale as they
    # already have the desired values from HunyuanConfig
