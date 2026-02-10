# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from fastvideo.configs.models import EncoderConfig
from fastvideo.configs.models.encoders import (
    BaseEncoderOutput,
    CLIPTextConfig,
    T5Config,
)
from fastvideo.configs.models.dits.sd3 import SD3DiTConfig
from fastvideo.configs.models.vaes.autoencoder_kl import AutoencoderKLVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig, preprocess_text


def _sd35_text_postprocess(outputs: BaseEncoderOutput) -> torch.Tensor:
    assert outputs.last_hidden_state is not None
    return outputs.last_hidden_state


@dataclass
class SD35Config(PipelineConfig):

    scheduler_arch: str = "FlowMatchEulerDiscreteScheduler"
    transformer_arch: str = "SD3Transformer2DModel"
    vae_arch: str = "AutoencoderKL"
    text_encoder_archs: tuple[str, ...] = (
        "CLIPTextModelWithProjection",
        "CLIPTextModelWithProjection",
        "T5EncoderModel",
    )
    tokenizer_archs: tuple[str, ...] = (
        "CLIPTokenizer",
        "CLIPTokenizer",
        "T5TokenizerFast",
    )

    dit_config: SD3DiTConfig = field(default_factory=SD3DiTConfig)
    vae_config: AutoencoderKLVAEConfig = field(
        default_factory=AutoencoderKLVAEConfig)

    embedded_cfg_scale: float = 0.0
    flow_shift: float | None = None

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda:
        (CLIPTextConfig(), CLIPTextConfig(), T5Config()))
    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda:
        (preprocess_text, preprocess_text, preprocess_text))
    postprocess_text_funcs: tuple[
        Callable[[BaseEncoderOutput], torch.Tensor],
        ...] = field(default_factory=lambda:
                     (_sd35_text_postprocess, _sd35_text_postprocess,
                      _sd35_text_postprocess))

    dit_precision: str = "bf16"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("fp32", "fp32", "bf16"))
