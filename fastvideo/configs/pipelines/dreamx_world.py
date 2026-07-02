# SPDX-License-Identifier: Apache-2.0
"""DreamX-World-5B-Cam FastVideo model configuration helpers."""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits.dreamx_world import (DreamXWorldARArchConfig, DreamXWorldARConfig,
                                                        DreamXWorldArchConfig, DreamXWorldConfig)
from fastvideo.configs.models.encoders import BaseEncoderOutput, T5Config
from fastvideo.configs.models.encoders.t5 import T5ArchConfig
from fastvideo.configs.models.vaes import WanVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.configs.pipelines.wan import LucyEditDevConfig, t5_postprocess_text


def make_dreamx_world_5b_cam_dit_config() -> DreamXWorldConfig:
    """Return the DreamX-World DiT config matching DreamX-World-5B-Cam."""
    return DreamXWorldConfig(arch_config=DreamXWorldArchConfig(
        num_attention_heads=24,
        attention_head_dim=128,
        in_channels=48,
        out_channels=48,
        ffn_dim=14336,
        num_layers=30,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        add_control_adapter=True,
        cam_method="prope",
        attn_compress=1,
        cam_self_attn_layers=None,
    ))


def make_dreamx_world_5b_ar_dit_config() -> DreamXWorldARConfig:
    """Return the DreamX-World-5B autoregressive causal DiT config."""
    return DreamXWorldARConfig(arch_config=DreamXWorldARArchConfig(
        model_type="ti2v",
        num_attention_heads=24,
        attention_head_dim=128,
        in_channels=48,
        out_channels=48,
        ffn_dim=14336,
        num_layers=30,
        cross_attn_norm=True,
        qk_norm=True,
        add_control_adapter=True,
        cam_method="prope",
        attn_compress=4,
        cam_self_attn_layers=tuple(range(30)),
        local_attn_size=12,
        sink_size=3,
        num_frames_per_block=3,
    ))


def make_dreamx_world_5b_cam_vae_config() -> WanVAEConfig:
    """Return the Wan2.2 48-channel VAE config used by DreamX-World-5B-Cam."""
    return LucyEditDevConfig().vae_config


def make_dreamx_world_5b_cam_text_encoder_config() -> T5Config:
    """Return the UMT5-XXL text encoder config used by DreamX-World-5B-Cam."""
    return T5Config(
        arch_config=T5ArchConfig(
            vocab_size=256384,
            d_model=4096,
            d_kv=64,
            d_ff=10240,
            num_layers=24,
            num_decoder_layers=None,
            num_heads=64,
            relative_attention_num_buckets=32,
            dropout_rate=0.0,
            text_len=512,
            feed_forward_proj="gelu",
            is_encoder_decoder=False,
        ),
        prefix="umt5",
    )


@dataclass
class DreamXWorld5BCamPipelineConfig(PipelineConfig):
    """Pipeline config for the first-scope DreamX-World-5B-Cam mode."""

    dit_config: DiTConfig = field(default_factory=make_dreamx_world_5b_cam_dit_config)
    vae_config: VAEConfig = field(default_factory=make_dreamx_world_5b_cam_vae_config)
    text_encoder_configs: tuple[EncoderConfig,
                                ...] = field(default_factory=lambda: (make_dreamx_world_5b_cam_text_encoder_config(), ))
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor],
                                  ...] = field(default_factory=lambda: (t5_postprocess_text, ))
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16", ))
    flow_shift: float | None = 3.0
    ti2v_task: bool = True
    expand_timesteps: bool = True
    vae_tiling: bool = False
    vae_sp: bool = False
    vae_precision: str = "fp32"
    vae_decode_precision: str | None = "bf16"
    dit_precision: str = "bf16"

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
        self.dit_config.expand_timesteps = self.expand_timesteps


@dataclass
class DreamXWorld5BARPipelineConfig(DreamXWorld5BCamPipelineConfig):
    """Pipeline config for DreamX-World-5B autoregressive forcing."""

    dit_config: DiTConfig = field(default_factory=make_dreamx_world_5b_ar_dit_config)
    flow_shift: float | None = 5.0
    ti2v_task: bool = True
    is_causal: bool = True
    dmd_denoising_steps: tuple[int, ...] = (1000, 750, 500, 250)
    warp_denoising_step: bool = True
    context_noise: float = 0.1
    num_frames_per_block: int = 3
    color_correction_strength: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.dit_config.expand_timesteps = True
