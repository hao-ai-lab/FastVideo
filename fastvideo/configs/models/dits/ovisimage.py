# SPDX-License-Identifier: Apache-2.0
"""Configuration for OvisImageTransformer2DModel"""

from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig


def _is_double_block(n: str, m) -> bool:
    """Match transformer_blocks.{i} (double-stream blocks)."""
    return "transformer_blocks" in n and "single" not in n and str.isdigit(
        n.split(".")[-1])


def _is_single_block(n: str, m) -> bool:
    """Match single_transformer_blocks.{i} (single-stream blocks)."""
    return "single_transformer_blocks" in n and str.isdigit(n.split(".")[-1])


@dataclass
class OvisImageTransformer2DModelArchConfig(DiTArchConfig):
    """Architecture configuration for OvisImageTransformer2DModel."""

    # Core architecture
    hidden_size: int = 3072  # num_attention_heads * attention_head_dim = 24 * 128
    num_attention_heads: int = 24
    attention_head_dim: int = 128
    num_layers: int = 6  # Number of joint (double) layers
    num_single_layers: int = 27  # Number of single layers

    # Input/output configuration
    in_channels: int = 64
    out_channels: int | None = None  # Can be None, defaults to in_channels
    patch_size: int = 1

    # Dimensions
    joint_attention_dim: int = 2048  # Context dimension from text encoder
    axes_dims_rope: list[int] = field(default_factory=lambda: [16, 56, 56])

    # Legacy fields from base DiTArchConfig
    num_channels_latents: int = 16  # VAE latent channels (in_channels=64 is packed=16*4)

    # FSDP: shard double and single transformer blocks
    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [_is_double_block, _is_single_block])

    # Compile: same as FSDP for now
    _compile_conditions: list = field(
        default_factory=lambda: [_is_double_block, _is_single_block])

    # Weight name mapping: identity (native attrs match HF attrs)
    param_names_mapping: dict = field(default_factory=dict)
    reverse_param_names_mapping: dict = field(default_factory=dict)
    lora_param_names_mapping: dict = field(default_factory=dict)


@dataclass
class OvisImageTransformer2DModelConfig(DiTConfig):
    """Configuration for Ovis-Image DiT."""

    arch_config: DiTArchConfig = field(
        default_factory=OvisImageTransformer2DModelArchConfig)
    prefix: str = "OvisImage"
