# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits import CosmosVideoConfig
from fastvideo.configs.models.encoders import BaseEncoderOutput, T5LargeConfig
from fastvideo.configs.models.vaes import CosmosVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig


def t5_large_postprocess_text(outputs: BaseEncoderOutput) -> torch.Tensor:
    """Postprocess T5 Large text encoder outputs for Cosmos pipeline.
    
    Return raw last_hidden_state without truncation/padding.
    """
    hidden_state = outputs.last_hidden_state

    if hidden_state is None:
        raise ValueError("T5 Large outputs missing last_hidden_state")

    nan_count = torch.isnan(hidden_state).sum()
    if nan_count > 0:
        hidden_state = hidden_state.masked_fill(torch.isnan(hidden_state), 0.0)

    # Zero out embeddings beyond actual sequence length
    if outputs.attention_mask is not None:
        attention_mask = outputs.attention_mask
        lengths = attention_mask.sum(dim=1).cpu()
        for i, length in enumerate(lengths):
            hidden_state[i, length:] = 0

    return hidden_state


@dataclass
class CosmosConfig(PipelineConfig):
    """Configuration for Cosmos2 Video2World pipeline matching diffusers."""

    dit_config: DiTConfig = field(default_factory=CosmosVideoConfig)

    vae_config: VAEConfig = field(default_factory=CosmosVAEConfig)

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5LargeConfig(), ))
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor],
                                  ...] = field(default_factory=lambda:
                                               (t5_large_postprocess_text, ))

    dit_precision: str = "bf16"
    vae_precision: str = "fp16"
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("bf16", ))

    conditioning_strategy: str = "frame_replace"
    min_num_conditional_frames: int = 1
    max_num_conditional_frames: int = 2
    sigma_conditional: float = 0.0001
    sigma_data: float = 1.0
    state_ch: int = 16
    state_t: int = 24
    text_encoder_class: str = "T5"

    embedded_cfg_scale: int = 6
    flow_shift: float = 1.0

    def __post_init__(self):
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True

        self._vae_latent_dim = 16
