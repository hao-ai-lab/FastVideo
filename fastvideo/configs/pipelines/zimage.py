# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from fastvideo.configs.models import EncoderConfig
from fastvideo.configs.models.dits.zimage import ZImageDiTConfig
from fastvideo.configs.models.encoders import BaseEncoderOutput
from fastvideo.configs.models.encoders.qwen3 import Qwen3TextConfig
from fastvideo.configs.models.vaes.autoencoder_kl import AutoencoderKLVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig, preprocess_text


def _zimage_text_postprocess(outputs: BaseEncoderOutput) -> torch.Tensor:
    if outputs.hidden_states is None:
        raise RuntimeError("Z-Image requires Qwen3 hidden states")
    return outputs.hidden_states[-2]


@dataclass
class ZImagePipelineConfig(PipelineConfig):
    """Configuration for the native Z-Image text-to-image pipeline."""

    scheduler_arch: str = "FlowMatchEulerDiscreteScheduler"
    transformer_arch: str = "ZImageTransformer2DModel"
    vae_arch: str = "AutoencoderKL"
    text_encoder_archs: tuple[str, ...] = ("Qwen3Model", )
    tokenizer_archs: tuple[str, ...] = ("Qwen2Tokenizer", )

    dit_config: ZImageDiTConfig = field(default_factory=ZImageDiTConfig)
    vae_config: AutoencoderKLVAEConfig = field(default_factory=AutoencoderKLVAEConfig)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Qwen3TextConfig(chat_template_enable_thinking=True), ))
    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(default_factory=lambda: (preprocess_text, ))
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor],
                                  ...] = field(default_factory=lambda: (_zimage_text_postprocess, ))

    dit_precision: str = "bf16"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16", ))
    vae_tiling: bool = False
    vae_sp: bool = False

    embedded_cfg_scale: float = 0.0
    flow_shift: float | None = 3.0
    scheduler_step_in_fp32: bool = True
    scheduler_sigma_min: float = 0.0
    scheduler_use_reference_discrete_timesteps: bool = True
