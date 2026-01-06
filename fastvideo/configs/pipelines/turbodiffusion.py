# SPDX-License-Identifier: Apache-2.0
"""
TurboDiffusion pipeline configurations.

TurboDiffusion uses RCM (recurrent Consistency Model) scheduler with
SLA (Sparse-Linear Attention) for fast 1-4 step video generation.
"""
from dataclasses import dataclass, field

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits import WanVideoConfig
from fastvideo.configs.models.encoders import CLIPVisionConfig
from fastvideo.configs.models.vaes import WanVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.configs.pipelines.wan import t5_postprocess_text, T5Config, BaseEncoderOutput

import torch
from collections.abc import Callable


@dataclass
class TurboDiffusionT2VConfig(PipelineConfig):
    """Base configuration for TurboDiffusion T2V pipeline.
    
    Uses RCM scheduler with sigma_max=80 for 1-4 step generation.
    No boundary_ratio (single model, no switching).
    """
    # DiT
    dit_config: DiTConfig = field(default_factory=WanVideoConfig)
    # VAE
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False

    # Denoising stage
    flow_shift: float | None = 3.0
    
    # No boundary_ratio for T2V (single model)
    boundary_ratio: float | None = None

    # Text encoding stage
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5Config(), ))
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor],
                                  ...] = field(default_factory=lambda:
                                               (t5_postprocess_text, ))

    # Precision for each component
    precision: str = "bf16"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("fp32", ))

    # self-forcing params
    warp_denoising_step: bool = True

    def __post_init__(self):
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True
        # Ensure no boundary_ratio is set in dit_config
        self.dit_config.boundary_ratio = None


@dataclass
class TurboDiffusionT2V_1_3B_Config(TurboDiffusionT2VConfig):
    """Configuration for TurboDiffusion T2V 1.3B model."""
    pass


@dataclass
class TurboDiffusionT2V_14B_Config(TurboDiffusionT2VConfig):
    """Configuration for TurboDiffusion T2V 14B model.
    
    Uses same config as 1.3B but with higher flow_shift for 14B model.
    """
    flow_shift: float | None = 5.0


@dataclass
class TurboDiffusionI2VConfig(PipelineConfig):
    """Base configuration for TurboDiffusion I2V pipeline.
    
    Uses RCM scheduler with sigma_max=200 for 1-4 step generation.
    Uses boundary_ratio=0.9 for high-noise to low-noise model switching.
    """
    # DiT
    dit_config: DiTConfig = field(default_factory=WanVideoConfig)
    # VAE
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False

    # Denoising stage
    flow_shift: float | None = 5.0
    
    boundary_ratio: float | None = 0.9

    # Text encoding stage
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5Config(), ))
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor],
                                  ...] = field(default_factory=lambda:
                                               (t5_postprocess_text, ))

    # Image encoder for I2V
    image_encoder_config: EncoderConfig = field(
        default_factory=CLIPVisionConfig)
    image_encoder_precision: str = "fp32"

    # Precision for each component
    precision: str = "bf16"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("fp32", ))

    # self-forcing params
    warp_denoising_step: bool = True

    def __post_init__(self):
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
        self.dit_config.boundary_ratio = self.boundary_ratio


@dataclass
class TurboDiffusionI2V_A14B_Config(TurboDiffusionI2VConfig):
    """Configuration for TurboDiffusion I2V A14B model."""
    pass
