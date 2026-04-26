# SPDX-License-Identifier: Apache-2.0
"""PipelineConfig for the Stable Audio Open 1.0 text-to-audio pipeline.

All components are FastVideo-native (see REVIEW item 30):
- VAE: `fastvideo.models.vaes.oobleck.OobleckVAE`
- DiT: `fastvideo.models.dits.stable_audio.StableAudioDiT`
- Conditioner: `fastvideo.models.encoders.stable_audio_conditioner.StableAudioMultiConditioner`
  (T5-base + 2× NumberConditioner, vendored from upstream)
- Sampler: `k_diffusion.sampling.sample_dpmpp_3m_sde` (sampling library,
  not a model class — same function the official repo's `generate_diffusion_cond` uses)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from fastvideo.configs.models import VAEConfig
from fastvideo.configs.models.vaes import OobleckVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig


@dataclass
class StableAudioT2AConfig(PipelineConfig):
    """Stable Audio Open 1.0 text-to-audio."""

    # VAE — first-class FastVideo Oobleck.
    vae_config: VAEConfig = field(default_factory=OobleckVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False

    # Text encoder/tokenizer slots are unused — the native
    # `StableAudioMultiConditioner` owns its own T5 weights.  Override
    # the empty defaults from `PipelineConfig` so length validation
    # (text_encoder_configs == precisions == preprocess == postprocess)
    # passes with all zero-length tuples.
    text_encoder_configs: tuple = field(default_factory=tuple)
    preprocess_text_funcs: tuple = field(default_factory=tuple)
    postprocess_text_funcs: tuple = field(default_factory=tuple)

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
    text_encoder_precisions: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # T2A: the audio VAE always runs decode; encode is only needed
        # for audio-to-audio variants.
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
