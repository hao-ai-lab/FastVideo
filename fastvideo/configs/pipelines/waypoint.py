# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Waypoint-1-Small world model."""

from dataclasses import dataclass, field

import torch

from fastvideo.configs.models import DiTConfig, EncoderConfig
from fastvideo.configs.models.dits.waypoint_transformer import WaypointConfig
from fastvideo.configs.models.encoders import BaseEncoderOutput, T5Config
from fastvideo.configs.pipelines.base import PipelineConfig


def umt5_postprocess_text(outputs: BaseEncoderOutput) -> torch.Tensor:
    """Postprocess UMT5 encoder outputs for Waypoint.
    
    Waypoint expects prompt embeddings of shape [B, seq_len, 2048].
    """
    mask: torch.Tensor = outputs.attention_mask
    hidden_state: torch.Tensor = outputs.last_hidden_state
    seq_lens = mask.gt(0).sum(dim=1).long()
    
    # Pad/truncate to fixed length (512 tokens max)
    max_len = 512
    prompt_embeds = []
    for u, v in zip(hidden_state, seq_lens, strict=True):
        if v > max_len:
            prompt_embeds.append(u[:max_len])
        else:
            pad = u.new_zeros(max_len - v, u.size(1))
            prompt_embeds.append(torch.cat([u[:v], pad]))
    
    return torch.stack(prompt_embeds, dim=0)


@dataclass
class WaypointT2VConfig(PipelineConfig):
    """Configuration for Waypoint-1-Small text-to-video world model.
    
    Waypoint is an interactive world model that generates video frames
    conditioned on text prompts and controller inputs (mouse, keyboard, scroll).
    """
    
    # DiT configuration
    dit_config: DiTConfig = field(default_factory=WaypointConfig)
    
    # VAE is loaded from the model repo (WorldEngineVAE) via dynamic module import.
    vae_tiling: bool = False
    vae_sp: bool = False
    
    # Text encoding - uses UMT5-XL (loaded from model subfolder)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5Config(), ))
    postprocess_text_funcs: tuple = field(
        default_factory=lambda: (umt5_postprocess_text, ))
    
    # Precision settings
    precision: str = "bf16"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("fp32", ))
    
    # Waypoint-specific settings
    # Fixed sigma schedule (no flow shift)
    flow_shift: float | None = None
    scheduler_sigmas: list[float] = field(
        default_factory=lambda: [1.0, 0.8609585762023926, 0.729332447052002, 0.3205108940601349, 0.0]
    )
    
    # Interactive generation settings
    is_causal: bool = True
    base_fps: int = 60
    
    # Control input settings
    n_buttons: int = 256
    
    def __post_init__(self):
        # Waypoint doesn't use standard VAE loading
        pass

