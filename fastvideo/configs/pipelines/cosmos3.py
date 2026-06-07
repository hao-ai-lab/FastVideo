# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 pipeline configuration.

Reference of record: the official ``cosmos-framework`` / ``nvidia/Cosmos3-Nano``
checkpoint (``model_index.json``). Cosmos3 is structurally different from
Cosmos 2.5:

- Dual-pathway (UND + GEN) DiT lives entirely inside ``Cosmos3VFMTransformer``
  (``Cosmos3VideoConfig``).
- No separate text encoder — the Qwen3-VL-text backbone is inside the DiT, so
  ``text_encoder_configs`` is the empty tuple. The Qwen2 tokenizer is loaded as
  the ``text_tokenizer`` checkpoint module by the component loader.
- VAE is Wan2.2 ``AutoencoderKLWan`` (z_dim=48, scale_factor_spatial=16),
  configured by ``Cosmos3VAEConfig`` (the checkpoint's exact latents_mean/std).
- Scheduler is diffusers ``UniPCMultistepScheduler`` (flow_prediction,
  use_flow_sigmas), loaded from the checkpoint.
- T2I default ``flow_shift`` is 3.0 (set per-request by ``_set_flow_shift``);
  T2V/I2V use the engine-init default of 1.0 baked into this config.
"""
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits.cosmos3 import (Cosmos3ArchConfig, Cosmos3VideoConfig)
from fastvideo.configs.models.encoders import BaseEncoderOutput
from fastvideo.configs.models.vaes import Cosmos3VAEConfig  # Wan2.2 AutoencoderKLWan
from fastvideo.configs.pipelines.base import PipelineConfig


def _identity_preprocess_text(prompt: str) -> str:
    return prompt


@dataclass
class Cosmos3Config(PipelineConfig):
    """Configuration for the Cosmos3 video generation pipeline (T2V/I2V/T2I).

    Wires the framework-parity-verified Cosmos3 components: the native
    ``Cosmos3VideoConfig`` DiT, the Wan2.2 ``Cosmos3VAEConfig`` VAE, the Qwen2
    tokenizer (loaded as ``text_tokenizer``), and the UniPC scheduler.
    """

    dit_config: DiTConfig = field(default_factory=lambda: Cosmos3VideoConfig(arch_config=Cosmos3ArchConfig()))

    vae_config: VAEConfig = field(default_factory=Cosmos3VAEConfig)

    # No separate text encoder: Cosmos3LanguageModel lives inside the DiT.
    text_encoder_configs: tuple[EncoderConfig, ...] = field(default_factory=tuple)

    preprocess_text_funcs: tuple[Callable[[str], str],
                                 ...] = field(default_factory=lambda: (_identity_preprocess_text, ))
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor], ...] = field(default_factory=tuple)

    dit_precision: str = "bf16"
    vae_precision: str = "bf16"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=tuple)

    embedded_cfg_scale: float = 0.0
    # T2V/I2V engine-init flow_shift (framework text2video/image2video default);
    # T2I overrides to 3.0 per request via Cosmos3DenoisingStage._set_flow_shift.
    flow_shift: float = 10.0

    vae_tiling: bool = False
    vae_sp: bool = False

    def __post_init__(self):
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
