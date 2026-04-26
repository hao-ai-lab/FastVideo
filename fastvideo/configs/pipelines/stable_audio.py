# SPDX-License-Identifier: Apache-2.0
"""PipelineConfig for the Stable Audio Open 1.0 text-to-audio pipeline.

This pipeline composes:
- Text encoder: HF `T5EncoderModel` (T5-base) — reused via standard
  FastVideo `T5Config`.
- Tokenizer: `T5TokenizerFast` from the same repo.
- Projection model: `diffusers.StableAudioProjectionModel` (custom number
  embedder + projection head). **Reused from diffusers** — first-class
  port is a follow-up (REVIEW item 30).
- Scheduler: `diffusers.CosineDPMSolverMultistepScheduler` (math-only,
  no learnable weights). **Reused from diffusers**.
- Transformer (DiT): `diffusers.StableAudioDiTModel` (24-layer DiT with
  GQA 24:12, head_dim=64). **Reused from diffusers** — first-class port
  is a follow-up (REVIEW item 30).
- VAE: FastVideo's first-class `OobleckVAE` (see
  `fastvideo/models/vaes/oobleck.py`).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from fastvideo.configs.models import EncoderConfig, VAEConfig
from fastvideo.configs.models.encoders import BaseEncoderOutput, T5Config
from fastvideo.configs.models.vaes import OobleckVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig


def t5_passthrough_postprocess(outputs: BaseEncoderOutput) -> torch.Tensor:
    """Stable Audio uses raw T5 last_hidden_state with attention_mask;
    no padding/trim needed at the postprocess stage (the projection
    model + masking happen inside the conditioning stage).
    """
    return outputs.last_hidden_state


@dataclass
class StableAudioT2AConfig(PipelineConfig):
    """Stable Audio Open 1.0 text-to-audio."""

    # VAE — first-class FastVideo Oobleck.
    vae_config: VAEConfig = field(default_factory=OobleckVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False

    # Text encoder — T5-base. The model_index lists T5EncoderModel; we
    # reuse FastVideo's existing T5 config (the encoder side is
    # identical to T5).
    text_encoder_configs: tuple[EncoderConfig, ...] = field(default_factory=lambda: (T5Config(), ))
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor],
                                  ...] = field(default_factory=lambda: (t5_passthrough_postprocess, ))

    # Pipeline-level defaults (sourced from
    # https://huggingface.co/stabilityai/stable-audio-open-1.0).
    num_inference_steps: int = 100
    guidance_scale: float = 7.0
    audio_end_in_s: float = 10.0  # default short clip; full max is ~47.55s
    audio_start_in_s: float = 0.0
    sampling_rate: int = 44100
    audio_channels: int = 2

    # Precisions.
    precision: str = "fp32"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("fp32", ))

    def __post_init__(self) -> None:
        # T2A: the audio VAE always runs decode; encode is only needed
        # for audio-to-audio variants. The VAEConfig contract has these
        # flags but the OobleckVAE (and its lazy wrapper) loads both
        # encoder and decoder regardless.
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
