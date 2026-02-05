# SPDX-License-Identifier: Apache-2.0
# Copied and adapted from: https://github.com/sglang-ai/sglang
from dataclasses import dataclass, field
from typing import Tuple

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig
from fastvideo.logger import init_logger

logger = init_logger(__name__)


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
    axes_dims_rope: Tuple[int, ...] = (32, 32, 32, 32)  # RoPE dimensions per axis (match diffusers Flux2)
    rope_theta: int = 2000  # Base frequency for RoPE (match diffusers Flux2)
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

    def update_from_weight_keys(self, all_keys: set[str]) -> None:
        """Infer num_layers and num_single_layers from checkpoint weight keys so the model is built with the same number of blocks as the weights."""
        if not all_keys:
            return
        num_layers = 0
        num_single_layers = 0
        for k in all_keys:
            if "single_transformer_blocks." not in k and "transformer_blocks." in k:
                parts = k.split("transformer_blocks.")[-1].split(".")
                if parts[0].isdigit():
                    num_layers = max(num_layers, int(parts[0]) + 1)
            if "single_transformer_blocks." in k:
                parts = k.split("single_transformer_blocks.")[-1].split(".")
                if parts[0].isdigit():
                    num_single_layers = max(num_single_layers, int(parts[0]) + 1)
        if num_layers > 0:
            self.num_layers = num_layers
            logger.info("Inferred num_layers=%s from checkpoint keys", num_layers)
        if num_single_layers > 0:
            self.num_single_layers = num_single_layers
            logger.info("Inferred num_single_layers=%s from checkpoint keys", num_single_layers)
        if num_layers > 0 or num_single_layers > 0:
            self.__post_init__()


@dataclass
class Flux2Config(DiTConfig):
    """Configuration for Flux2 transformer model."""
    
    arch_config: DiTArchConfig = field(default_factory=Flux2ArchConfig)
    
    prefix: str = "Flux"
