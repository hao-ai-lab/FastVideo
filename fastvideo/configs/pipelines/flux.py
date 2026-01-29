# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits import FluxConfig
from fastvideo.configs.models.encoders import (BaseEncoderOutput,
                                               CLIPTextConfig, T5Config)
from fastvideo.configs.pipelines.base import PipelineConfig


def clip_preprocess_text(prompt: str) -> str:
    return prompt


def clip_postprocess_text(outputs: BaseEncoderOutput) -> torch.Tensor:
    return outputs.pooler_output


def t5_preprocess_text(prompt: str) -> str:
    return prompt


def t5_postprocess_text(outputs: BaseEncoderOutput) -> torch.Tensor:
    return outputs.last_hidden_state


@dataclass
class FluxT2IConfig(PipelineConfig):
    """Configuration for Flux text-to-image pipeline."""

    dit_config: DiTConfig = field(default_factory=FluxConfig)
    vae_config: VAEConfig = field(default_factory=VAEConfig)
    flow_shift: float | None = 3.0

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (CLIPTextConfig(), T5Config()))
    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (clip_preprocess_text, t5_preprocess_text))
    postprocess_text_funcs: tuple[
        Callable[[BaseEncoderOutput], torch.Tensor],
        ...] = field(default_factory=lambda:
                     (clip_postprocess_text, t5_postprocess_text))

    dit_precision: str = "bf16"
    vae_precision: str = "bf16"
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("bf16", "bf16"))

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True
