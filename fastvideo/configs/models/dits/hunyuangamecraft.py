# SPDX-License-Identifier: Apache-2.0
"""
Configuration for HunyuanGameCraft transformer model.

HunyuanGameCraft extends HunyuanVideo with:
1. CameraNet for camera/action conditioning
2. 33 input channels (16 latent + 16 gt_latent + 1 mask)
3. Mask-based conditioning for autoregressive generation
"""
from dataclasses import dataclass, field

import torch

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig


def is_double_block(n: str, m) -> bool:
    return "double" in n and str.isdigit(n.split(".")[-1])


def is_single_block(n: str, m) -> bool:
    return "single" in n and str.isdigit(n.split(".")[-1])


def is_refiner_block(n: str, m) -> bool:
    return "refiner" in n and str.isdigit(n.split(".")[-1])


def is_txt_in(n: str, m) -> bool:
    return n.split(".")[-1] == "txt_in"


def is_camera_net(n: str, m) -> bool:
    return "camera_net" in n


@dataclass
class HunyuanGameCraftArchConfig(DiTArchConfig):
    """Architecture config for HunyuanGameCraft transformer."""

    # Version field for compatibility with saved config.json
    _fastvideo_version: str = "0.1.0"

    # Camera net flag (for config.json compatibility)
    camera_net: bool = True

    _fsdp_shard_conditions: list = field(
        default_factory=lambda:
        [is_double_block, is_single_block, is_refiner_block, is_camera_net])

    _compile_conditions: list = field(
        default_factory=lambda: [is_double_block, is_single_block, is_txt_in])

    # Parameter names mapping from official checkpoint to FastVideo naming
    # GameCraft weights are already close to FastVideo format with minor adjustments
    param_names_mapping: dict = field(
        default_factory=lambda: {
            # MLP naming: fc1 -> fc_in, fc2 -> fc_out
            r"^(.*)\.img_mlp\.fc1\.(.*)$":
            r"\1.img_mlp.fc_in.\2",
            r"^(.*)\.img_mlp\.fc2\.(.*)$":
            r"\1.img_mlp.fc_out.\2",
            r"^(.*)\.txt_mlp\.fc1\.(.*)$":
            r"\1.txt_mlp.fc_in.\2",
            r"^(.*)\.txt_mlp\.fc2\.(.*)$":
            r"\1.txt_mlp.fc_out.\2",

            # Single block MLP naming
            r"^single_blocks\.(\d+)\.mlp\.fc1\.(.*)$":
            r"single_blocks.\1.mlp.fc_in.\2",
            r"^single_blocks\.(\d+)\.mlp\.fc2\.(.*)$":
            r"single_blocks.\1.mlp.fc_out.\2",

            # Token refiner naming
            r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.(.*)$":
            r"txt_in.refiner_blocks.\1.\2",

            # Vector in naming
            r"^vector_in\.in_layer\.(.*)$":
            r"vector_in.fc_in.\1",
            r"^vector_in\.out_layer\.(.*)$":
            r"vector_in.fc_out.\1",

            # Time embedder naming
            r"^time_in\.mlp\.0\.(.*)$":
            r"time_in.mlp.fc_in.\1",
            r"^time_in\.mlp\.2\.(.*)$":
            r"time_in.mlp.fc_out.\1",

            # Guidance embedder naming (if present)
            r"^guidance_in\.mlp\.0\.(.*)$":
            r"guidance_in.mlp.fc_in.\1",
            r"^guidance_in\.mlp\.2\.(.*)$":
            r"guidance_in.mlp.fc_out.\1",

            # Final layer adaLN modulation
            r"^final_layer\.adaLN_modulation\.1\.(.*)$":
            r"final_layer.adaLN_modulation.linear.\1",

            # Refiner block MLP naming
            r"^txt_in\.refiner_blocks\.(\d+)\.mlp\.fc1\.(.*)$":
            r"txt_in.refiner_blocks.\1.mlp.fc_in.\2",
            r"^txt_in\.refiner_blocks\.(\d+)\.mlp\.fc2\.(.*)$":
            r"txt_in.refiner_blocks.\1.mlp.fc_out.\2",

            # Camera net weights are already correctly named
        })

    # Reverse mapping for saving checkpoints
    reverse_param_names_mapping: dict = field(default_factory=lambda: {})

    # Model architecture parameters
    # patch_size can be int or tuple - if tuple, it's [T, H, W]
    patch_size: int | tuple[int, int, int] = 2
    patch_size_t: int = 1
    in_channels: int = 33  # 16 latent + 16 gt_latent + 1 mask
    out_channels: int = 16
    num_attention_heads: int = 24
    attention_head_dim: int = 128
    mlp_ratio: float = 4.0
    num_layers: int = 20  # Double stream blocks
    num_single_layers: int = 40  # Single stream blocks
    num_refiner_layers: int = 2
    rope_axes_dim: tuple[int, int, int] = (16, 56, 56)
    guidance_embeds: bool = False  # GameCraft doesn't use guidance
    dtype: torch.dtype | None = None
    text_embed_dim: int = 4096  # LLaMA-3 hidden size
    pooled_projection_dim: int = 768  # CLIP pooled output dim
    rope_theta: int = 256
    qk_norm: str = "rms_norm"

    # Camera net parameters
    camera_in_channels: int = 6  # Plücker coordinates
    camera_downscale_coef: int = 8
    camera_out_channels: int = 16

    # Layers to exclude from LoRA
    exclude_lora_layers: list[str] = field(
        default_factory=lambda:
        ["img_in", "txt_in", "time_in", "vector_in", "camera_net"])

    def __post_init__(self):
        super().__post_init__()
        self.hidden_size: int = self.attention_head_dim * self.num_attention_heads
        self.num_channels_latents: int = 16  # Output is 16 channels

        # Convert patch_size list to tuple if needed (from JSON deserialization)
        if isinstance(self.patch_size, list):
            self.patch_size = tuple(self.patch_size)

        # Convert rope_axes_dim list to tuple if needed
        if isinstance(self.rope_axes_dim, list):
            self.rope_axes_dim = tuple(self.rope_axes_dim)


@dataclass
class HunyuanGameCraftConfig(DiTConfig):
    """Full config for HunyuanGameCraft model."""

    arch_config: DiTArchConfig = field(
        default_factory=HunyuanGameCraftArchConfig)

    prefix: str = "HunyuanGameCraft"
