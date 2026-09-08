# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits import Cosmos25VideoConfig
from fastvideo.configs.models.dits.cosmos2_5 import (
    Cosmos25ArchConfig,
    Cosmos25_14BArchConfig,
    Cosmos25_14BVideoConfig,
)
from fastvideo.configs.models.encoders import BaseEncoderOutput
from fastvideo.configs.models.encoders.cosmos_predict_text_encoder import CosmosPredictTextEncoderConfig, CosmosPredictTextEncoderArchConfig
from fastvideo.configs.models.vaes import Cosmos25VAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig


def _identity_preprocess_text(prompt: str) -> str:
    return prompt


def cosmos_predict_postprocess_text(outputs: BaseEncoderOutput) -> torch.Tensor:
    # Just return hidden states unmodified. The text encoder handles its own logic if needed.
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise ValueError("Cosmos Predict postprocess requires outputs.hidden_states")
    return hidden_states


@dataclass
class CosmosPredictConfig(PipelineConfig):
    """Configuration for Cosmos Predict (Text-to-Video/Video-to-Video) generation pipeline."""

    dit_config: DiTConfig = field(default_factory=lambda: Cosmos25VideoConfig(arch_config=Cosmos25ArchConfig(
        num_attention_heads=16,
        attention_head_dim=128,
        in_channels=16,
        out_channels=16,
        num_layers=28,
        patch_size=[1, 2, 2],
        max_size=[128, 240, 240],
        rope_scale=[1.0, 3.0, 3.0],
        text_embed_dim=1024,
        mlp_ratio=4.0,
        adaln_lora_dim=256,
        use_adaln_lora=True,
        concat_padding_mask=True,
        extra_pos_embed_type=None,
        use_crossattn_projection=True,
        rope_enable_fps_modulation=False,
        qk_norm="rms_norm",
        use_condition_mask=False, # Cosmos Predict does not concat condition mask
    )))

    vae_config: VAEConfig = field(default_factory=Cosmos25VAEConfig)

    text_encoder_configs: tuple[EncoderConfig, ...] = field(default_factory=lambda: (CosmosPredictTextEncoderConfig(
        arch_config=CosmosPredictTextEncoderArchConfig()), ))

    preprocess_text_funcs: tuple[Callable[[str], str],
                                 ...] = field(default_factory=lambda: (_identity_preprocess_text, ))
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor],
                                  ...] = field(default_factory=lambda: (cosmos_predict_postprocess_text, ))

    dit_precision: str = "bf16"
    vae_precision: str = "bf16"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16", ))

    embedded_cfg_scale: float = 0.0
    flow_shift: float = 5.0

    vae_tiling: bool = False
    vae_sp: bool = False

    def __post_init__(self):
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
        self._vae_latent_dim = 16


@dataclass
class CosmosPredict14BConfig(CosmosPredictConfig):
    """Configuration for Cosmos Predict 14B pipeline."""

    dit_config: DiTConfig = field(default_factory=lambda: Cosmos25_14BVideoConfig(arch_config=Cosmos25_14BArchConfig(
        num_attention_heads=40,
        attention_head_dim=128,
        in_channels=16,
        out_channels=16,
        num_layers=36,
        patch_size=[1, 2, 2],
        max_size=[128, 240, 240],
        rope_scale=[1.0, 3.0, 3.0],
        text_embed_dim=1024,
        mlp_ratio=4.0,
        adaln_lora_dim=256,
        use_adaln_lora=True,
        concat_padding_mask=True,
        extra_pos_embed_type=None,
        use_crossattn_projection=True,
        rope_enable_fps_modulation=False,
        qk_norm="rms_norm",
        use_condition_mask=False,
    )))

