# SPDX-License-Identifier: Apache-2.0
"""`PipelineConfig` for Stable Audio Open 1.0."""
from __future__ import annotations

from dataclasses import dataclass, field

from fastvideo.configs.models import VAEConfig
from fastvideo.configs.models.vaes import OobleckVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig


@dataclass
class StableAudioT2AConfig(PipelineConfig):
    """Stable Audio Open 1.0 pipeline config."""

    vae_config: VAEConfig = field(default_factory=OobleckVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False

    # `StableAudioMultiConditioner` owns its own T5; zero out the
    # parent's text-encoder slots so the length-equality validator passes.
    text_encoder_configs: tuple = field(default_factory=tuple)
    preprocess_text_funcs: tuple = field(default_factory=tuple)
    postprocess_text_funcs: tuple = field(default_factory=tuple)

    num_inference_steps: int = 100
    guidance_scale: float = 7.0
    audio_end_in_s: float = 10.0  # short-clip default
    audio_start_in_s: float = 0.0
    sampling_rate: int = 44100
    audio_channels: int = 2
    # Stable Audio Open 1.0 was trained at a fixed 2,097,152-sample
    # window (= 2097152 / 44100 ≈ 47.55s). Anything past this is
    # silently truncated by the post-decode slice — validate up-front.
    max_audio_duration_s: float = 2097152 / 44100

    # Match the official `stable_audio_tools` defaults (`model_half=True`
    # in `run_gradio.py`), which loads the DiT, VAE, and T5 in fp16 and
    # wraps T5 forward in `autocast(fp16)`. fp16 is also a hard
    # requirement for FlashAttention-2 / FA-3.
    precision: str = "fp16"
    vae_precision: str = "fp16"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # A2A needs encode; load both halves for either path.
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
