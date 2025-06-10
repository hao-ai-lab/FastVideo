# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Callable, Tuple

import torch

from fastvideo.v1.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.v1.configs.models.dits import WanVideoConfig
from fastvideo.v1.configs.models.encoders import (BaseEncoderOutput,
                                                  CLIPVisionConfig, T5Config)
from fastvideo.v1.configs.models.vaes import WanVAEConfig
from fastvideo.v1.configs.pipelines.base import PipelineConfig


def t5_postprocess_text(outputs: BaseEncoderOutput) -> torch.tensor:
    mask: torch.tensor = outputs.attention_mask
    hidden_state: torch.tensor = outputs.last_hidden_state
    seq_lens = mask.gt(0).sum(dim=1).long()
    assert torch.isnan(hidden_state).sum() == 0
    prompt_embeds = [u[:v] for u, v in zip(hidden_state, seq_lens)]
    prompt_embeds_tensor: torch.tensor = torch.stack([
        torch.cat([u, u.new_zeros(512 - u.size(0), u.size(1))])
        for u in prompt_embeds
    ],
                                                     dim=0)
    return prompt_embeds_tensor


@dataclass
class WanT2V480PConfig(PipelineConfig):
    """Base configuration for Wan T2V 1.3B pipeline architecture."""

    # WanConfig-specific parameters with defaults
    # DiT
    dit_config: DiTConfig = field(default_factory=WanVideoConfig)
    # VAE
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False

    # Denoising stage
    flow_shift: int = 3

    # Text encoding stage
    text_encoder_configs: Tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5Config(), ))
    postprocess_text_funcs: Tuple[Callable[[BaseEncoderOutput], torch.tensor],
                                  ...] = field(default_factory=lambda:
                                               (t5_postprocess_text, ))

    # Precision for each component
    precision: str = "bf16"
    vae_precision: str = "fp16"
    text_encoder_precisions: Tuple[str, ...] = field(
        default_factory=lambda: ("fp32", ))

    # WanConfig-specific added parameters

    def __post_init__(self):
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True


@dataclass
class WanT2V720PConfig(WanT2V480PConfig):
    """Base configuration for Wan T2V 14B 720P pipeline architecture."""

    # WanConfig-specific parameters with defaults

    # Denoising stage
    flow_shift: int = 5


@dataclass
class WanI2V480PConfig(WanT2V480PConfig):
    """Base configuration for Wan I2V 14B 480P pipeline architecture."""

    # WanConfig-specific parameters with defaults

    # Precision for each component
    image_encoder_config: EncoderConfig = field(
        default_factory=CLIPVisionConfig)
    image_encoder_precision: str = "fp32"

    def __post_init__(self):
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True


@dataclass
class WanI2V720PConfig(WanI2V480PConfig):
    """Base configuration for Wan I2V 14B 720P pipeline architecture."""

    # WanConfig-specific parameters with defaults

    # Denoising stage
    flow_shift: int = 5
