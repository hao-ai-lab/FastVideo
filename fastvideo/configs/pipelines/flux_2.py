# SPDX-License-Identifier: Apache-2.0
# Copied and adapted from: https://github.com/sglang-ai/sglang
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from fastvideo.configs.models import DiTConfig, EncoderConfig, VAEConfig
from fastvideo.configs.models.dits.flux_2 import Flux2Config
from fastvideo.configs.models.encoders import BaseEncoderOutput
from fastvideo.configs.models.encoders.base import EncoderArchConfig
from fastvideo.configs.models.encoders.qwen3 import Qwen3TextConfig
from fastvideo.configs.models.vaes.flux2vae import Flux2VAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig, preprocess_text


@dataclass
class Flux2PipelineConfig(PipelineConfig):
    """Configuration for Flux2 image generation pipeline."""
    
    # Flux2-specific parameters
    embedded_cfg_scale: float = 4.0
    
    # DiT configuration
    dit_config: DiTConfig = field(default_factory=Flux2Config)
    dit_precision: str = "bf16"
    
    # VAE configuration
    vae_config: VAEConfig = field(default_factory=Flux2VAEConfig)
    vae_precision: str = "fp32"
    vae_tiling: bool = False  # Flux2 is image model, disable tiling by default
    vae_sp: bool = False
    
    # Text encoder configuration (Flux2 uses Mistral/Qwen)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (EncoderConfig(),)
    )
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("bf16",)
    )
    
    # Default postprocess function (can be overridden)
    @staticmethod
    def default_postprocess_text(outputs: BaseEncoderOutput) -> torch.Tensor:
        """Default text postprocessing for Flux2."""
        return outputs.last_hidden_state
    
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor], ...] = field(
        default_factory=lambda: (Flux2PipelineConfig.default_postprocess_text,)
    )


def flux2_klein_postprocess_text(outputs: BaseEncoderOutput) -> torch.Tensor:
    """Klein postprocess: hidden states from layers 9, 18, 27 (Qwen3)."""
    hidden_states_layers: list[int] = [9, 18, 27]
    if outputs.hidden_states is None:
        raise ValueError("Flux2 Klein requires output_hidden_states=True from text encoder")
    out = torch.stack([outputs.hidden_states[k] for k in hidden_states_layers], dim=1)
    batch_size, num_channels, seq_len, hidden_dim = out.shape
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(
        batch_size, seq_len, num_channels * hidden_dim
    )
    return prompt_embeds


@dataclass
class Flux2KleinEncoderArchConfig(EncoderArchConfig):
    """Encoder arch config for Flux2 Klein (Qwen3); needs hidden states for layers 9, 18, 27."""
    output_hidden_states: bool = True


@dataclass
class Flux2KleinTextEncoderConfig(EncoderConfig):
    """Text encoder config for Flux2 Klein (Qwen3)."""
    arch_config: EncoderArchConfig = field(default_factory=Flux2KleinEncoderArchConfig)


@dataclass
class Flux2KleinPipelineConfig(Flux2PipelineConfig):
    """Configuration for Flux2 Klein (distilled, 4-step, no guidance)."""
    embedded_cfg_scale: float = 0.0  # No guidance for distilled Klein
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Qwen3TextConfig(),)
    )
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("bf16",)
    )
    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (preprocess_text,)
    )
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor], ...] = field(
        default_factory=lambda: (flux2_klein_postprocess_text,)
    )
