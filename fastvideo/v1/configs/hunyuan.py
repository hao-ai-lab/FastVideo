from dataclasses import dataclass
from fastvideo.v1.configs.base import BaseConfig


@dataclass
class HunyuanConfig(BaseConfig):
    """Base configuration for HunYuan pipeline architecture."""

    # HunyuanConfig-specific parameters with defaults
    embedded_cfg_scale: int = 6
    flow_shift: int = 7

    # Override some BaseConfig defaults
    num_inference_steps: int = 50


@dataclass
class FastHunyuanConfig(HunyuanConfig):
    """Configuration specifically optimized for FastHunyuan weights."""

    # Override HunyuanConfig defaults
    num_inference_steps: int = 6
    flow_shift: int = 17

    # No need to re-specify guidance_scale or embedded_cfg_scale as they
    # already have the desired values from HunyuanConfig
