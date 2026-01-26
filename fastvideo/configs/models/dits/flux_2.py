# SPDX-License-Identifier: Apache-2.0
# Copied and adapted from: https://github.com/sglang-ai/sglang
from dataclasses import dataclass, field
from typing import Tuple

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class Flux2ArchConfig(DiTArchConfig):
    """Architecture configuration for Flux2 transformer model."""
    
    # Flux2-specific architecture parameters
    patch_size: int = 1
    in_channels: int = 64
    out_channels: int | None = None
    num_layers: int = 19  # Number of double-stream transformer blocks
    num_single_layers: int = 38  # Number of single-stream transformer blocks
    attention_head_dim: int = 128
    num_attention_heads: int = 24
    joint_attention_dim: int = 4096  # Dimension for text encoder output
    timestep_guidance_channels: int = 256  # Dimension for timestep embedding
    mlp_ratio: float = 3.0
    axes_dims_rope: Tuple[int, int, int] = (16, 56, 56)  # RoPE dimensions for each axis
    rope_theta: int = 10000  # Base frequency for RoPE
    eps: float = 1e-6
    guidance_embeds: bool = True  # Whether to use guidance embeddings
    
    # Parameter name mapping for loading HuggingFace checkpoints
    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"transformer\.(\w*)\.(.*)$": r"\1.\2",
        }
    )

    def __post_init__(self):
        super().__post_init__()
        self.out_channels = self.out_channels or self.in_channels
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels


@dataclass
class Flux2Config(DiTConfig):
    """Configuration for Flux2 transformer model."""
    
    arch_config: DiTArchConfig = field(default_factory=Flux2ArchConfig)
    
    prefix: str = "Flux"
