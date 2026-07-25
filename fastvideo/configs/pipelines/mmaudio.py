# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for the native MMAudio video-to-audio port."""

from __future__ import annotations

from dataclasses import dataclass, field

from fastvideo.configs.models import DiTConfig, EncoderConfig, ModelConfig
from fastvideo.configs.models.audio import (BigVGANV2ArchConfig, BigVGANV2Config, MMAudioVAEArchConfig,
                                            MMAudioVAEConfig)
from fastvideo.configs.models.dits import MMAudioTransformerConfig, get_mmaudio_transformer_config
from fastvideo.configs.models.encoders import (
    MMAudioDFNCLIPTextConfig,
    MMAudioDFNCLIPVisionConfig,
    MMAudioSynchformerConfig,
)
from fastvideo.configs.pipelines.base import PipelineConfig


@dataclass
class MMAudioV2AConfig(PipelineConfig):
    """MMAudio large-44k-v2 inference defaults.

    The published demo moves every module to bfloat16. Keeping the same
    per-component precision here is important: condition features seed the
    complete flow trajectory, so silently encoding them in fp32 changes the
    generated waveform even when the transformer weights are identical.
    """

    dit_config: DiTConfig = field(default_factory=MMAudioTransformerConfig)
    dit_precision: str = "bf16"

    text_encoder_configs: tuple[EncoderConfig, ...] = field(default_factory=lambda: (MMAudioDFNCLIPTextConfig(), ))
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16", ))
    image_encoder_configs: tuple[EncoderConfig, ...] = field(default_factory=lambda: (
        MMAudioDFNCLIPVisionConfig(),
        MMAudioSynchformerConfig(),
    ))
    image_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16", "bf16"))

    audio_decoder_config: ModelConfig = field(default_factory=MMAudioVAEConfig)
    audio_decoder_precision: str = "bf16"
    # Offline training preprocessing uses the VAE encoder in fp32, matching
    # the published MMAudio feature extraction recipe. Inference remains
    # decoder-only bf16 through ``audio_decoder_precision`` above.
    audio_encoder_precision: str = "fp32"
    vocoder_config: ModelConfig = field(default_factory=BigVGANV2Config)
    vocoder_precision: str = "bf16"

    # Published large_44k_v2 default sequence contract. The official demo
    # supports other durations, although quality can drop far away from the
    # eight-second training duration.
    duration_s: float = 8.0
    max_audio_duration_s: float | None = None
    sampling_rate: int = 44100
    spectrogram_frame_rate: int = 512
    latent_downsample_rate: int = 2
    clip_frame_rate: int = 8
    sync_frame_rate: int = 25
    sync_segment_size: int = 16
    sync_segment_stride: int = 8
    sync_downsample_rate: int = 2
    clip_image_size: int = 384
    sync_image_size: int = 224
    clip_batch_size_multiplier: int = 40
    sync_batch_size_multiplier: int = 40

    num_inference_steps: int = 25
    guidance_scale: float = 4.5
    vae_tiling: bool = False
    vae_sp: bool = False


@dataclass
class MMAudioSmall16kV2AConfig(MMAudioV2AConfig):
    """Official small 16 kHz MMAudio inference configuration."""

    dit_config: DiTConfig = field(default_factory=lambda: get_mmaudio_transformer_config("small_16k"))
    audio_decoder_config: ModelConfig = field(default_factory=lambda: MMAudioVAEConfig(arch_config=MMAudioVAEArchConfig(
        mode="16k",
        data_dim=80,
        embed_dim=20,
        hidden_dim=384,
    )))
    vocoder_config: ModelConfig = field(default_factory=lambda: BigVGANV2Config(arch_config=BigVGANV2ArchConfig(
        sample_rate=16000,
        num_mels=80,
    )))
    sampling_rate: int = 16000
    spectrogram_frame_rate: int = 256


@dataclass
class MMAudioSmall44kV2AConfig(MMAudioV2AConfig):
    dit_config: DiTConfig = field(default_factory=lambda: get_mmaudio_transformer_config("small_44k"))


@dataclass
class MMAudioMedium44kV2AConfig(MMAudioV2AConfig):
    dit_config: DiTConfig = field(default_factory=lambda: get_mmaudio_transformer_config("medium_44k"))


@dataclass
class MMAudioLarge44kV2AConfig(MMAudioV2AConfig):
    dit_config: DiTConfig = field(default_factory=lambda: get_mmaudio_transformer_config("large_44k"))


@dataclass
class MMAudioLarge44kV2PipelineConfig(MMAudioV2AConfig):
    dit_config: DiTConfig = field(default_factory=lambda: get_mmaudio_transformer_config("large_44k_v2"))


MMAUDIO_PIPELINE_CONFIGS = {
    "small_16k": MMAudioSmall16kV2AConfig,
    "small_44k": MMAudioSmall44kV2AConfig,
    "medium_44k": MMAudioMedium44kV2AConfig,
    "large_44k": MMAudioLarge44kV2AConfig,
    "large_44k_v2": MMAudioLarge44kV2PipelineConfig,
}
