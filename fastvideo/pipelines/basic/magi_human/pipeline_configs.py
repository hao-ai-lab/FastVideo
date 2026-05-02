# SPDX-License-Identifier: Apache-2.0
"""PipelineConfig for the daVinci-MagiHuman base text-to-AV pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits import MagiHumanVideoConfig
from fastvideo.configs.models.encoders import (
    BaseEncoderOutput,
    T5GemmaEncoderConfig,
)
from fastvideo.configs.models.vaes import OobleckVAEConfig, WanVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig


def t5gemma_postprocess_text(outputs: BaseEncoderOutput) -> torch.Tensor:
    """Return per-prompt last_hidden_state as a batched [B, L, D] tensor.

    MagiHuman pads/trims the embedding to a fixed length in its own
    `pad_or_trim` helper at pipeline time. Here we simply hand through
    whatever the tokenizer produced — the latent-prep stage is responsible
    for pad/trim so that the original context length can be preserved.
    """
    hidden = outputs.last_hidden_state
    assert torch.isnan(hidden).sum() == 0
    # Keep the shape the tokenizer emitted; the pipeline stage handles
    # pad-or-trim to t5_gemma_target_length=640.
    return hidden


@dataclass
class MagiHumanBaseConfig(PipelineConfig):
    """Base MagiHuman text-to-AV pipeline config (prompt → video + audio).

    MagiHuman's base model is a joint audio-visual generator. This config
    wires up both the video VAE (Wan 2.2 TI2V-5B) and the audio VAE
    (Stable Audio Open 1.0); the pipeline produces an mp4 with a muxed
    audio track. The framework's `WorkloadType` enum has no `T2AV`
    variant yet, so the registry entry uses `WorkloadType.T2V` as a
    placeholder.
    """

    # DiT
    dit_config: DiTConfig = field(default_factory=MagiHumanVideoConfig)
    # VAE — Wan 2.2 TI2V-5B. Diffusers `vae/config.json` drives arch_config
    # at load time, including z_dim=48 and scale_factor_temporal=4 /
    # scale_factor_spatial=16.
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False

    # Audio VAE — Stable Audio Open 1.0 (Oobleck), shared with the
    # standalone Stable Audio pipeline. Lazy-loaded from
    # `stabilityai/stable-audio-open-1.0` (HF gated, Apache 2.0).
    audio_vae_config: VAEConfig = field(default_factory=OobleckVAEConfig)

    # Denoising (flow-matching UniPC).
    flow_shift: float | None = 5.0

    # Text encoding
    text_encoder_configs: tuple[EncoderConfig, ...] = field(default_factory=lambda: (T5GemmaEncoderConfig(), ))
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor],
                                  ...] = field(default_factory=lambda: (t5gemma_postprocess_text, ))

    # Precisions — the DiT runs bf16 internally, the text encoder is
    # bf16-native, and the VAE decode path benefits from fp32 for long
    # sequences.
    precision: str = "bf16"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16", ))

    # MagiHuman-specific defaults surfaced for the pipeline stages. These
    # are pipeline-level knobs sourced from the upstream
    # `EvaluationConfig` / `DataProxyConfig` (not `ModelConfig`), so they
    # belong here, NOT on `MagiHumanArchConfig`.
    t5_gemma_target_length: int = 640
    fps: int = 25
    num_inference_steps: int = 32
    video_txt_guidance_scale: float = 5.0
    audio_txt_guidance_scale: float = 5.0
    cfg_number: int = 2

    # VAE / data-proxy knobs (were on ArchConfig before; moved here).
    vae_stride: tuple[int, int, int] = (4, 16, 16)
    z_dim: int = 48
    frame_receptive_field: int = 11
    coords_style: str = "v2"
    ref_audio_offset: int = 1000
    text_offset: int = 0

    # Video CFG step-dependent guidance: low-t steps use a relaxed scale.
    # Upstream daVinci-MagiHuman/inference/pipeline/video_generate.py:426
    # uses 5.0 for high-t and 2.0 for low-t with cutoff at t=500.
    video_guidance_high_t_threshold: int = 500
    video_guidance_low_t_value: float = 2.0

    def __post_init__(self) -> None:
        # Base text-to-AV does not need the VAE encoder (no reference-image
        # conditioning). Keep decoder only to save memory.
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True


@dataclass
class MagiHumanBaseI2VConfig(MagiHumanBaseConfig):
    """Base MagiHuman text+image-to-AV pipeline config.

    TI2V reuses the T2V DiT weights; the only pipeline-side difference is
    that a reference image is encoded with the Wan VAE and reinserted into
    the first video-latent frame before every denoise step.
    """

    image_conditioning: bool = True

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True


@dataclass
class MagiHumanDistillConfig(MagiHumanBaseConfig):
    """DMD-2 distilled MagiHuman text-to-AV pipeline config.

    Same arch as base (identical 331 keys, same shapes, same module tree),
    but trained via DMD-2 for 8-step inference without classifier-free
    guidance. Weights are stored in fp32 upstream; the conversion script's
    `--cast-bf16` flag reduces the checkpoint to ~30 GB on disk.
    """

    num_inference_steps: int = 8
    cfg_number: int = 1  # DMD distilled models skip CFG.
    # Lower flow_shift matches the distilled DMD schedule; if parity later
    # shows drift, measure against `scheduler_config.json` generated by the
    # conversion script for the distill subfolder.
    flow_shift: float | None = 5.0


@dataclass
class MagiHumanDistillI2VConfig(MagiHumanDistillConfig):
    """DMD-2 distilled MagiHuman text+image-to-AV pipeline config."""

    image_conditioning: bool = True

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True


@dataclass
class MagiHumanSR540pConfig(MagiHumanBaseConfig):
    """Two-stage MagiHuman base + SR-540p text-to-AV pipeline config."""

    noise_value: int = 220
    sr_audio_noise_scale: float = 0.7
    sr_num_inference_steps: int = 5
    sr_video_txt_guidance_scale: float = 3.5
    use_cfg_trick: bool = True
    cfg_trick_start_frame: int = 13
    cfg_trick_value: float = 2.0
    # Upstream example/sr_540p uses sr_height=512, sr_width=896. Despite the
    # marketing name, these are the VAE/patch-aligned dimensions actually run.
    sr_height: int = 512
    sr_width: int = 896
    sr_local_attn_layers: tuple[int, ...] = ()


@dataclass
class MagiHumanSR540pI2VConfig(MagiHumanSR540pConfig):
    """Two-stage MagiHuman base + SR-540p text+image-to-AV config."""

    image_conditioning: bool = True

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True


_SR_1080P_LOCAL_ATTN_LAYERS: tuple[int, ...] = (
    0,
    1,
    2,
    4,
    5,
    6,
    8,
    9,
    10,
    12,
    13,
    14,
    16,
    17,
    18,
    20,
    21,
    22,
    24,
    25,
    26,
    28,
    29,
    30,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
)


@dataclass
class MagiHumanSR1080pConfig(MagiHumanSR540pConfig):
    """Two-stage MagiHuman base + SR-1080p text-to-AV pipeline config."""

    sr_height: int = 1080
    sr_width: int = 1920
    sr_local_attn_layers: tuple[int, ...] = _SR_1080P_LOCAL_ATTN_LAYERS


@dataclass
class MagiHumanSR1080pI2VConfig(MagiHumanSR1080pConfig):
    """Two-stage MagiHuman base + SR-1080p text+image-to-AV config."""

    image_conditioning: bool = True

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
