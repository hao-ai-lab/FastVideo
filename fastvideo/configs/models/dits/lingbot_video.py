# SPDX-License-Identifier: Apache-2.0
"""Architecture configuration for LingBot-Video Dense and MoE DiTs."""

from __future__ import annotations

from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig
from fastvideo.platforms import AttentionBackendEnum


def _is_lingbot_video_block(name: str, module: object) -> bool:
    """Select top-level transformer blocks for FSDP and compilation."""
    del module
    parts = name.split(".")
    return len(parts) == 2 and parts[0] == "blocks" and parts[1].isdigit()


@dataclass
class LingBotVideoArchConfig(DiTArchConfig):
    """One-to-one representation of the released transformer config JSON."""

    _fsdp_shard_conditions: list = field(default_factory=lambda: [_is_lingbot_video_block])
    _supported_attention_backends: tuple[AttentionBackendEnum, ...] = (
        AttentionBackendEnum.TORCH_SDPA,
        AttentionBackendEnum.FLASH_ATTN,
    )
    param_names_mapping: dict = field(default_factory=lambda: {r"^(.*)$": r"\1"})

    patch_size: tuple[int, int, int] = (1, 2, 2)
    in_channels: int = 16
    out_channels: int = 16
    hidden_size: int = 2048
    num_attention_heads: int = 16
    depth: int = 24
    intermediate_size: int = 6144
    text_dim: int = 2560
    freq_dim: int = 256
    norm_eps: float = 1e-6
    rope_theta: float = 256.0
    axes_dims: tuple[int, int, int] = (32, 48, 48)
    axes_lens: tuple[int, int, int] = (8192, 1024, 1024)
    qkv_bias: bool = False
    out_bias: bool = True
    patch_embed_bias: bool = True
    timestep_mlp_bias: bool = True
    num_experts: int = 0
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 512
    decoder_sparse_step: int = 1
    mlp_only_layers: tuple[int, ...] = ()
    n_shared_experts: int | None = None
    score_func: str = "sigmoid"
    norm_topk_prob: bool = True
    n_group: int | None = None
    topk_group: int | None = None
    routed_scaling_factor: float = 1.0

    def __post_init__(self) -> None:
        """Populate FastVideo loader fields from the released architecture."""
        super().__post_init__()
        self.num_channels_latents = self.in_channels
        self.attention_head_dim = self.hidden_size // self.num_attention_heads


@dataclass
class LingBotVideoConfig(DiTConfig):
    """FastVideo component configuration for LingBot-Video transformers."""

    arch_config: DiTArchConfig = field(default_factory=LingBotVideoArchConfig)
