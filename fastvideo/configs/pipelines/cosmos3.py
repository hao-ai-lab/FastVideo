# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 pipeline configuration.

Port reference: vllm-omni PR #3454 (https://github.com/vllm-project/vllm-omni/pull/3454)
HEAD ``8536f5b1421f``. Cosmos3 is structurally different from Cosmos 2.5:

- Dual-stack DiT (UND + GEN) lives entirely inside ``Cosmos3VFMTransformer``.
- No separate text encoder — ``Cosmos3LanguageModel`` is inside the DiT, so
  ``text_encoder_configs`` is the empty tuple. The tokenizer is loaded by the
  pipeline's ``__init__`` from the checkpoint's ``text_tokenizer/`` subfolder;
  that is an instance-level concern, not a config-level concern.
- VAE is ``DistributedAutoencoderKLWan`` (same as Cosmos 2.5), so the
  ``Cosmos25VAEConfig`` is reused as-is.
- T2I default ``flow_shift`` is 3.0 (set per-request by ``_set_flow_shift``);
  T2V/I2V use the engine-init default of 1.0 baked into this config.
"""
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits.cosmos3 import (Cosmos3ArchConfig, Cosmos3VideoConfig)
from fastvideo.configs.models.encoders import BaseEncoderOutput
from fastvideo.configs.models.vaes import Cosmos25VAEConfig  # Cosmos3 reuses the Wan-based VAE
from fastvideo.configs.pipelines.base import PipelineConfig


def _identity_preprocess_text(prompt: str) -> str:
    return prompt


@dataclass
class Cosmos3Config(PipelineConfig):
    """Configuration for the Cosmos3 video generation pipeline (T2V/I2V/T2I).

    Tier A scope: architectural skeleton. Real weights from ``nvidia/Cosmos3-Nano``
    are not yet published; this config is shaped to be activatable when they
    land without further config changes.
    """

    dit_config: DiTConfig = field(default_factory=lambda: Cosmos3VideoConfig(arch_config=Cosmos3ArchConfig()))

    vae_config: VAEConfig = field(default_factory=Cosmos25VAEConfig)

    # No separate text encoder: Cosmos3LanguageModel lives inside the DiT.
    text_encoder_configs: tuple[EncoderConfig, ...] = field(default_factory=tuple)

    preprocess_text_funcs: tuple[Callable[[str], str],
                                 ...] = field(default_factory=lambda: (_identity_preprocess_text, ))
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor], ...] = field(default_factory=tuple)

    dit_precision: str = "bf16"
    vae_precision: str = "bf16"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=tuple)

    embedded_cfg_scale: float = 0.0
    flow_shift: float = 1.0  # T2V/I2V engine-init; T2I uses 3.0 set per-request

    vae_tiling: bool = False
    vae_sp: bool = False

    def __post_init__(self):
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
