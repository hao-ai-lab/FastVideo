# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from fastvideo.configs.models import EncoderConfig
from fastvideo.configs.models.dits.flux import FluxDiTConfig
from fastvideo.configs.models.encoders import (
    BaseEncoderOutput,
    CLIPTextConfig,
    T5LargeConfig,
)
from fastvideo.configs.models.vaes.autoencoder_kl import AutoencoderKLVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig, preprocess_text


def _flux_clip_pooled_postprocess(outputs: BaseEncoderOutput) -> torch.Tensor:
    """CLIP branch for FLUX: Diffusers uses pooled prompt embeddings only."""
    if outputs.pooler_output is None:
        raise RuntimeError(
            "FLUX CLIP conditioning requires pooler_output. Ensure the CLIP text encoder returns pooled features."
        )
    return outputs.pooler_output


def _flux_t5_sequence_postprocess(outputs: BaseEncoderOutput) -> torch.Tensor:
    if outputs.last_hidden_state is None:
        raise RuntimeError("FLUX T5 conditioning requires last_hidden_state.")
    return outputs.last_hidden_state


@dataclass
class FluxPipelineConfig(PipelineConfig):
    """Pipeline layout for Diffusers FLUX.1-dev (CLIP + T5 + packed DiT + FlowMatch)."""

    scheduler_arch: str = "FlowMatchEulerDiscreteScheduler"
    transformer_arch: str = "FluxTransformer2DModel"
    vae_arch: str = "AutoencoderKL"
    text_encoder_archs: tuple[str, ...] = ("CLIPTextModel", "T5EncoderModel")
    tokenizer_archs: tuple[str, ...] = ("CLIPTokenizer", "T5TokenizerFast")

    dit_config: FluxDiTConfig = field(default_factory=FluxDiTConfig)
    vae_config: AutoencoderKLVAEConfig = field(default_factory=AutoencoderKLVAEConfig)

    embedded_cfg_scale: float = 3.5
    flow_shift: float | None = None

    text_encoder_configs: tuple[EncoderConfig, ...] = field(default_factory=lambda: (CLIPTextConfig(), T5LargeConfig()))
    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (preprocess_text, preprocess_text)
    )
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor], ...] = field(
        default_factory=lambda: (_flux_clip_pooled_postprocess, _flux_t5_sequence_postprocess)
    )

    dit_precision: str = "bf16"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("fp32", "bf16"))

    def __post_init__(self) -> None:
        te_cfgs = list(self.text_encoder_configs)
        if len(te_cfgs) >= 1:
            te_cfgs[0].tokenizer_kwargs.setdefault("padding", "max_length")
            te_cfgs[0].tokenizer_kwargs.setdefault("max_length", 77)
            te_cfgs[0].tokenizer_kwargs.setdefault("truncation", True)
            te_cfgs[0].tokenizer_kwargs.setdefault("return_tensors", "pt")
        if len(te_cfgs) >= 2:
            cap = 512
            te_cfgs[1].tokenizer_kwargs["max_length"] = min(
                int(te_cfgs[1].tokenizer_kwargs.get("max_length", cap)), cap
            )
            te_cfgs[1].tokenizer_kwargs.setdefault("padding", "max_length")
            te_cfgs[1].tokenizer_kwargs.setdefault("truncation", True)
            te_cfgs[1].tokenizer_kwargs.setdefault("return_tensors", "pt")
