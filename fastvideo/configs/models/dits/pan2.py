# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig
from fastvideo.platforms import AttentionBackendEnum


def is_transformer_block(n: str, m) -> bool:
    return "transformer_blocks" in n and str.isdigit(n.split(".")[-1])


def is_refiner_block(n: str, m) -> bool:
    return "context_refiner" in n and str.isdigit(n.split(".")[-1])


@dataclass
class PAN2ArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_transformer_block, is_refiner_block])
    _supported_attention_backends: tuple[AttentionBackendEnum, ...] = (
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.TORCH_SDPA,
    )

    # Diffusers PAN2Transformer3DModel keys -> native keys. The per-stream Q/K/V projections are fused on load.
    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^x_embedder\.(.*)$": r"x_embedder.proj.\1",
            r"^time_embedder\.linear_1\.(.*)$": r"time_embedder.mlp.fc_in.\1",
            r"^time_embedder\.linear_2\.(.*)$": r"time_embedder.mlp.fc_out.\1",
            r"^context_embedder\.0\.(.*)$": r"context_norm.\1",
            r"^context_embedder\.1\.(.*)$": r"context_proj.\1",
            r"^context_refiner\.(\d+)\.attn\.to_q\.(.*)$": (r"context_refiner.\1.attn.to_qkv.\2", 0, 3),
            r"^context_refiner\.(\d+)\.attn\.to_k\.(.*)$": (r"context_refiner.\1.attn.to_qkv.\2", 1, 3),
            r"^context_refiner\.(\d+)\.attn\.to_v\.(.*)$": (r"context_refiner.\1.attn.to_qkv.\2", 2, 3),
            r"^context_refiner\.(\d+)\.attn\.to_out\.0\.(.*)$": r"context_refiner.\1.attn.to_out.\2",
            r"^context_refiner\.(\d+)\.ff\.net\.0\.proj\.(.*)$": r"context_refiner.\1.ff.fc_in.\2",
            r"^context_refiner\.(\d+)\.ff\.net\.2\.(.*)$": r"context_refiner.\1.ff.fc_out.\2",
            r"^transformer_blocks\.(\d+)\.attn\.to_q\.(.*)$": (r"transformer_blocks.\1.attn.to_qkv.\2", 0, 3),
            r"^transformer_blocks\.(\d+)\.attn\.to_k\.(.*)$": (r"transformer_blocks.\1.attn.to_qkv.\2", 1, 3),
            r"^transformer_blocks\.(\d+)\.attn\.to_v\.(.*)$": (r"transformer_blocks.\1.attn.to_qkv.\2", 2, 3),
            r"^transformer_blocks\.(\d+)\.attn\.add_q_proj\.(.*)$":
            (r"transformer_blocks.\1.attn.add_qkv_proj.\2", 0, 3),
            r"^transformer_blocks\.(\d+)\.attn\.add_k_proj\.(.*)$":
            (r"transformer_blocks.\1.attn.add_qkv_proj.\2", 1, 3),
            r"^transformer_blocks\.(\d+)\.attn\.add_v_proj\.(.*)$":
            (r"transformer_blocks.\1.attn.add_qkv_proj.\2", 2, 3),
            r"^transformer_blocks\.(\d+)\.attn\.to_out\.0\.(.*)$": r"transformer_blocks.\1.attn.to_out.\2",
            r"^transformer_blocks\.(\d+)\.(ff|ff_context)\.net\.0\.proj\.(.*)$": r"transformer_blocks.\1.\2.fc_in.\3",
            r"^transformer_blocks\.(\d+)\.(ff|ff_context)\.net\.2\.(.*)$": r"transformer_blocks.\1.\2.fc_out.\3",
        })

    reverse_param_names_mapping: dict = field(default_factory=lambda: {})

    patch_size: tuple[int, ...] = (1, 2, 2)
    in_channels: int = 97
    out_channels: int = 48
    num_attention_heads: int = 32
    attention_head_dim: int = 128
    num_layers: int = 44
    num_refiner_layers: int = 4
    mlp_ratio: float = 8.0
    context_mlp_ratio: float = 4.0
    refiner_mlp_ratio: float = 4.0
    text_embed_dim: int = 4096
    rope_axes_dim: tuple[int, ...] = (16, 56, 56)
    rope_theta: float = 256.0
    # Frame positions are rescaled to the frame rate the rotary embedding was trained at (16 fps of 24 fps).
    rope_temporal_scale: float = 16 / 24
    # PAN2 runs its text stream in the DiT dtype.
    cast_prompt_embeds_to_dit_dtype: bool = True
    exclude_lora_layers: list[str] = field(
        default_factory=lambda: ["x_embedder", "time_embedder", "context_norm", "context_proj", "context_refiner"])

    def __post_init__(self):
        super().__post_init__()
        self.patch_size = tuple(self.patch_size)
        self.rope_axes_dim = tuple(self.rope_axes_dim)
        self.hidden_size: int = self.attention_head_dim * self.num_attention_heads
        self.num_channels_latents: int = self.out_channels


@dataclass
class PAN2VideoConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=PAN2ArchConfig)

    prefix: str = "PAN2"
