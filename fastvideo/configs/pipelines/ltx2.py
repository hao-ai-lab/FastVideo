# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from fastvideo.configs.models import (DiTConfig, EncoderConfig, ModelConfig,
                                      LTX2AudioDecoderConfig, LTX2VocoderConfig,
                                      VAEConfig)
from fastvideo.configs.models.dits import LTX2VideoConfig
from fastvideo.configs.models.encoders import BaseEncoderOutput, LTX2GemmaConfig
from fastvideo.configs.models.vaes import LTX2VAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig, preprocess_text


def ltx2_postprocess_text(outputs: BaseEncoderOutput) -> torch.Tensor:
    return outputs.last_hidden_state


@dataclass
class LTX2T2VConfig(PipelineConfig):
    """Configuration for LTX-2 T2V pipeline."""

    dit_config: DiTConfig = field(default_factory=LTX2VideoConfig)
    vae_config: VAEConfig = field(default_factory=LTX2VAEConfig)
    vae_tiling: bool = True
    vae_sp: bool = False

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (LTX2GemmaConfig(), ))
    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (preprocess_text, ))
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor],
                                  ...] = field(default_factory=lambda:
                                               (ltx2_postprocess_text, ))

    dit_precision: str = "bf16"
    vae_precision: str = "bf16"
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("bf16", ))

    audio_decoder_config: ModelConfig = field(
        default_factory=LTX2AudioDecoderConfig)
    vocoder_config: ModelConfig = field(default_factory=LTX2VocoderConfig)
    audio_decoder_precision: str = "bf16"
    vocoder_precision: str = "bf16"

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True
