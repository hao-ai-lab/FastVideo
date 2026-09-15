# SPDX-License-Identifier: Apache-2.0
"""Typed component wiring for Helios-Distilled T2V inference."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import html

import ftfy
import regex as re
import torch

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits.helios import HeliosConfig
from fastvideo.configs.models.encoders import BaseEncoderOutput, T5Config
from fastvideo.configs.models.encoders.base import TextEncoderArchConfig
from fastvideo.configs.models.encoders.t5 import T5ArchConfig
from fastvideo.configs.models.vaes import WanVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig


@dataclass
class HeliosT5ArchConfig(T5ArchConfig):

    def __post_init__(self) -> None:
        super().__post_init__()
        self.tokenizer_kwargs["padding"] = "max_length"


@dataclass
class HeliosT5Config(T5Config):
    arch_config: TextEncoderArchConfig = field(default_factory=HeliosT5ArchConfig)


def helios_preprocess_text(prompt: str) -> str:
    text = ftfy.fix_text(prompt)
    text = html.unescape(html.unescape(text)).strip()
    return re.sub(r"\s+", " ", text).strip()


def helios_postprocess_text(output: BaseEncoderOutput) -> torch.Tensor:
    if output.last_hidden_state is None or output.attention_mask is None:
        raise ValueError("Helios UMT5 output requires hidden states and attention mask")
    hidden_states = output.last_hidden_state
    sequence_lengths = output.attention_mask.gt(0).sum(dim=1).long()
    if torch.isnan(hidden_states).any():
        raise ValueError("Helios UMT5 produced NaN hidden states")
    trimmed = [hidden[:min(int(length), 512)] for hidden, length in zip(hidden_states, sequence_lengths, strict=True)]
    return torch.stack(
        [torch.cat([hidden, hidden.new_zeros(512 - hidden.shape[0], hidden.shape[1])]) for hidden in trimmed],
        dim=0,
    )


def make_helios_text_encoder_config() -> T5Config:
    return HeliosT5Config(
        arch_config=HeliosT5ArchConfig(
            architectures=["UMT5EncoderModel"],
            vocab_size=256384,
            d_model=4096,
            d_kv=64,
            d_ff=10240,
            num_layers=24,
            num_decoder_layers=24,
            num_heads=64,
            relative_attention_num_buckets=32,
            relative_attention_max_distance=128,
            dropout_rate=0.1,
            layer_norm_epsilon=1e-6,
            feed_forward_proj="gated-gelu",
            is_encoder_decoder=True,
            use_cache=True,
            text_len=512,
        ),
        prefix="umt5",
    )


@dataclass
class HeliosPipelineConfig(PipelineConfig):
    dit_config: DiTConfig = field(default_factory=HeliosConfig)
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    text_encoder_configs: tuple[EncoderConfig,
                                ...] = field(default_factory=lambda: (make_helios_text_encoder_config(), ))
    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(default_factory=lambda: (helios_preprocess_text, ))
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor],
                                  ...] = field(default_factory=lambda: (helios_postprocess_text, ))
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16", ))
    dit_precision: str = "bf16"
    vae_precision: str = "fp32"
    vae_decode_precision: str | None = "fp32"
    vae_tiling: bool = False
    vae_sp: bool = False
    flow_shift: float | None = None

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True
