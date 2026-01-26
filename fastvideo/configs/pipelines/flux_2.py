# SPDX-License-Identifier: Apache-2.0
# Copied and adapted from: https://github.com/sglang-ai/sglang
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits.flux_2 import Flux2Config
from fastvideo.configs.models.encoders import BaseEncoderOutput
from fastvideo.configs.models.vaes.flux2vae import Flux2VAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig


@dataclass
class Flux2PipelineConfig(PipelineConfig):
    """Configuration for Flux2 image generation pipeline."""
    
    # Flux2-specific parameters
    embedded_cfg_scale: float = 4.0
    
    # DiT configuration
    dit_config: DiTConfig = field(default_factory=Flux2Config)
    dit_precision: str = "bf16"
    
    # VAE configuration
    vae_config: VAEConfig = field(default_factory=Flux2VAEConfig)
    vae_precision: str = "fp32"
    vae_tiling: bool = False  # Flux2 is image model, disable tiling by default
    vae_sp: bool = False
    
    # Text encoder configuration (Flux2 uses Mistral/Qwen)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (EncoderConfig(),)
    )
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("bf16",)
    )
    
    # Default postprocess function (can be overridden)
    @staticmethod
    def default_postprocess_text(outputs: BaseEncoderOutput) -> torch.Tensor:
        """Default text postprocessing for Flux2."""
        return outputs.last_hidden_state
    
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor], ...] = field(
        default_factory=lambda: (Flux2PipelineConfig.default_postprocess_text,)
    )
