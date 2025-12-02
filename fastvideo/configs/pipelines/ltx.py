# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from collections.abc import Callable

import torch
from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits import LTXVideoConfig
from fastvideo.configs.models.vaes import LTXVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig

from fastvideo.configs.models.encoders import (BaseEncoderOutput, T5Config)


def ltx_postprocess_text(outputs: BaseEncoderOutput) -> torch.Tensor:
    """LTX uses the full T5 output without padding."""
    return outputs.last_hidden_state


def ltx_t5_config() -> T5Config:
    config = T5Config()
    config.text_len = 128
    config.tokenizer_kwargs["max_length"] = 128
    return config


@dataclass
class LTXConfig(PipelineConfig):
    """Configuration for LTX Text-to-Video and Image-to-Video pipelines."""

    # TODO: fix all of the configs so it's exact match

    # # Text encoder configuration
    # text_encoder_configs: tuple[EncoderConfig, ...] = field(
    #     # todo: set max length later
    #     #def ltx_t5_config():
    #     #     config = T5Config()
    #     #     config.tokenizer_kwargs["max_length"] = 128
    #     #     return config

    #     # @dataclass
    #     # class LTXConfig(PipelineConfig):
    #     #     text_encoder_configs: tuple[EncoderConfig, ...] = field(
    #     #         default_factory=lambda: (ltx_t5_config(), ))
    #     default_factory=lambda: (T5Config(), ))

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (ltx_t5_config(), ))
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor],
                                  ...] = field(default_factory=lambda:
                                               (ltx_postprocess_text, ))
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("bf16", ))

    # DiT configuration
    dit_config: DiTConfig = field(default_factory=LTXVideoConfig)
    dit_precision: str = "bf16"

    # VAE configuration
    vae_config: VAEConfig = field(default_factory=LTXVAEConfig)
    vae_precision: str = "bf16"
    vae_tiling: bool = False
    vae_sp: bool = False

    # LTX architecture parameters
    vae_spatial_compression_ratio: int = 32
    vae_temporal_compression_ratio: int = 8
    transformer_spatial_patch_size: int = 1
    transformer_temporal_patch_size: int = 1

    # Scheduler adaptive shift parameters
    base_image_seq_len: int = 256
    max_image_seq_len: int = 4096
    base_shift: float = 0.5
    max_shift: float = 1.15

    # Generation defaults
    default_width: int = 704
    default_height: int = 480
    default_num_frames: int = 161
    default_frame_rate: int = 25
    default_num_inference_steps: int = 50
    guidance_scale: float = 7.5
    guidance_rescale: float = 0.0

    embedded_cfg_scale = None

    # Stochastic decoding (optional)
    default_decode_timestep: float = 0.0
    default_decode_noise_scale: float | None = None
    vae_latents_mean: torch.Tensor | None = None  # For normalization
    vae_latents_std: torch.Tensor | None = None  # For normalization
    # Mode flags (set at runtime based on usage)
    ltx_mode: bool = True
    ltx_i2v_mode: bool = False  # Set to True when doing image-to-video in InputValidationStage

    # I2V specific parameters
    conditioning_frame_indices: list[int] = field(default_factory=lambda: [0])
    use_conditioning_mask: bool = False  # TODO: can add later

    def __post_init__(self):
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True

        # TODO: load differently for each config
        # Text-to-Video: Only needs the decoder (to decode latents to video)
        # Image-to-Video: Needs both encoder (to encode input image) and decoder
        # @dataclass
        # class LTXT2VConfig(LTXConfig):
        #     def __post_init__(self):
        #         super().__post_init__()
        #         self.vae_config.load_encoder = False
        #         self.vae_config.load_decoder = True

        # @dataclass
        # class LTXI2VConfig(LTXConfig):
        #     def __post_init__(self):
        #         super().__post_init__()
        #         self.vae_config.load_encoder = True
        #         self.vae_config.load_decoder = True

        if hasattr(self.dit_config, 'patch_size'):
            self.transformer_spatial_patch_size = self.dit_config.patch_size
        if hasattr(self.dit_config, 'patch_size_t'):
            self.transformer_temporal_patch_size = self.dit_config.patch_size_t
