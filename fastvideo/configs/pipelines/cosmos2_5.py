# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits import Cosmos25VideoConfig
from fastvideo.configs.models.dits.cosmos2_5 import Cosmos25ArchConfig
from fastvideo.configs.models.encoders import BaseEncoderOutput
from fastvideo.configs.models.encoders.reason1 import Reason1Config, Reason1ArchConfig
from fastvideo.configs.models.vaes import Cosmos25VAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig


def _identity_preprocess_text(prompt: str) -> str:
    return prompt


def reason1_postprocess_text(outputs: BaseEncoderOutput) -> torch.Tensor:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise ValueError("Reason1 postprocess requires outputs.hidden_states")

    hs = list(hidden_states)[1:]
    normed = []
    for h in hs:
        h = h.float()
        h = (h - h.mean(dim=-1, keepdim=True)) / (h.std(dim=-1, keepdim=True) +
                                                  1e-8)
        normed.append(h)
    return torch.cat(normed, dim=-1).to(hidden_states[0].dtype)


@dataclass
class Cosmos25Config(PipelineConfig):
    """Configuration for Cosmos 2.5 (Predict2.5) video generation pipeline."""

    dit_config: DiTConfig = field(default_factory=lambda: Cosmos25VideoConfig(
        arch_config=Cosmos25ArchConfig(
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
        )))

    vae_config: VAEConfig = field(default_factory=Cosmos25VAEConfig)

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Reason1Config(arch_config=Reason1ArchConfig(
            embedding_concat_strategy="full_concat")), ))

    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (_identity_preprocess_text, ))
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor],
                                  ...] = field(default_factory=lambda:
                                               (reason1_postprocess_text, ))

    dit_precision: str = "bf16"
    vae_precision: str = "bf16"
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("bf16", ))

    embedded_cfg_scale: float = 0.0
    flow_shift: float = 5.0

    vae_tiling: bool = False
    vae_sp: bool = False

    def __post_init__(self):
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
        self._vae_latent_dim = 16
