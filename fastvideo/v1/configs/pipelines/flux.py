from dataclasses import dataclass, field
from typing import Callable, Tuple

import torch

from fastvideo.v1.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.v1.configs.models.dits import FluxImageConfig
from fastvideo.v1.configs.models.encoders import (BaseEncoderOutput,
                                                  CLIPTextConfig, T5Config)
from fastvideo.v1.configs.models.vaes import ImageVAEConfig
from fastvideo.v1.configs.pipelines.base import PipelineConfig


def t5_preprocess_text(prompt: str) -> str:
    return prompt


def t5_postprocess_text(outputs: BaseEncoderOutput) -> torch.tensor:
    hidden_state: torch.tensor = outputs.last_hidden_state
    assert torch.isnan(hidden_state).sum() == 0
    prompt_embeds_tensor: torch.tensor = torch.stack([
        torch.cat([u, u.new_zeros(512 - u.size(0), u.size(1))])
        for u in hidden_state
    ],
                                                     dim=0)
    return prompt_embeds_tensor


def clip_preprocess_text(prompt: str) -> str:
    return prompt


def clip_postprocess_text(outputs: BaseEncoderOutput) -> torch.tensor:
    pooler_output: torch.tensor = outputs.pooler_output
    return pooler_output


@dataclass
class FluxConfig(PipelineConfig):
    """Base configuration for Flux pipeline architecture."""

    # FluxConfig-specific parameters with defaults
    # DiT
    dit_config: DiTConfig = field(default_factory=FluxImageConfig)
    # VAE
    vae_config: VAEConfig = field(default_factory=ImageVAEConfig)
    # Denoising stage
    embedded_cfg_scale: float = 3.5

    # Text encoding stage
    text_encoder_configs: Tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (CLIPTextConfig(), T5Config()))
    preprocess_text_funcs: Tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (clip_preprocess_text, t5_preprocess_text))
    postprocess_text_funcs: Tuple[
        Callable[[BaseEncoderOutput], torch.tensor],
        ...] = field(default_factory=lambda:
                     (clip_postprocess_text, t5_postprocess_text))

    # Precision for each component
    precision: str = "bf16"
    vae_precision: str = "fp16"
    text_encoder_precisions: Tuple[str, ...] = field(
        default_factory=lambda: ("bf16", "bf16"))

    def __post_init__(self):
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True
