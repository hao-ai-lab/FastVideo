# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Callable

import torch

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits.davinci_magihuman import (
    DaVinciMagiHumanArchConfig, DaVinciMagiHumanConfig)
from fastvideo.configs.models.encoders.base import BaseEncoderOutput
from fastvideo.configs.models.encoders.t5gemma import T5GemmaConfig
from fastvideo.configs.models.vaes.davinci_vae import DaVinciVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig


def _identity_preprocess_text(prompt: str) -> str:
    return prompt


def _identity_postprocess_text(outputs: BaseEncoderOutput) -> torch.Tensor:
    """Return last_hidden_state directly (shape: B, N_t, 3584)."""
    return outputs.last_hidden_state


@dataclass
class DaVinciMagiHumanPipelineConfig(PipelineConfig):
    """Configuration for daVinci-MagiHuman text-to-video pipeline."""

    dit_config: DiTConfig = field(
        default_factory=lambda: DaVinciMagiHumanConfig(
            arch_config=DaVinciMagiHumanArchConfig()))

    vae_config: VAEConfig = field(default_factory=DaVinciVAEConfig)

    # T5Gemma-9B text encoder (google/t5gemma-9b — gated on HuggingFace)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5GemmaConfig(),))

    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (_identity_preprocess_text,))
    postprocess_text_funcs: tuple[
        Callable[[BaseEncoderOutput], torch.Tensor], ...] = field(
        default_factory=lambda: (_identity_postprocess_text,))

    dit_precision: str = "bf16"
    vae_precision: str = "bf16"
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("bf16",))

    # Flow matching with shift=5.0 (same as Cosmos 2.5 and Wan)
    flow_shift: float = 5.0
    embedded_cfg_scale: float = 0.0

    vae_tiling: bool = False
    vae_sp: bool = False

    def __post_init__(self):
        self.vae_config.load_encoder = False  # inference: decode only
        self.vae_config.load_decoder = True
        self._vae_latent_dim = 48  # z_dim
