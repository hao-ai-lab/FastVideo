from dataclasses import dataclass

from fastvideo.v1.configs.base import BaseConfig


@dataclass
class HunyuanConfig(BaseConfig):
    """Base configuration for HunYuan pipeline architecture."""

    # HunyuanConfig-specific parameters with defaults
    # Denoising stage
    embedded_cfg_scale: int = 6
    flow_shift: int = 7
    num_inference_steps: int = 50

    # Text encoding stage
    hidden_state_skip_layer: int = 2
    text_len: int = 256

    # Precision for each component
    precision: str = "bf16"
    vae_precision: str = "fp16"
    text_encoder_precision: str = "fp16"

    # HunyuanConfig-specific added parameters
    # Secondary text encoder
    text_encoder_precision_2: str = "fp16"
    text_len_2: int = 77


@dataclass
class FastHunyuanConfig(HunyuanConfig):
    """Configuration specifically optimized for FastHunyuan weights."""

    # Override HunyuanConfig defaults
    num_inference_steps: int = 6
    flow_shift: int = 17

    # No need to re-specify guidance_scale or embedded_cfg_scale as they
    # already have the desired values from HunyuanConfig
