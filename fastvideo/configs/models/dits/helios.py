# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig
from fastvideo.platforms import AttentionBackendEnum


def _is_transformer_block(name: str, module) -> bool:
    del module
    return name.startswith("blocks.") and name.split(".")[-1].isdigit()


@dataclass
class HeliosArchConfig(DiTArchConfig):
    """Architecture fields for Helios-Distilled's history-aware DiT."""

    _fsdp_shard_conditions: list = field(default_factory=lambda: [_is_transformer_block])
    _supported_attention_backends: tuple[AttentionBackendEnum, ...] = (
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.TORCH_SDPA,
    )
    param_names_mapping: dict = field(default_factory=dict)
    reverse_param_names_mapping: dict = field(default_factory=dict)
    lora_param_names_mapping: dict = field(default_factory=dict)

    patch_size: tuple[int, int, int] = (1, 2, 2)
    num_attention_heads: int = 40
    attention_head_dim: int = 128
    in_channels: int = 16
    out_channels: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 13824
    num_layers: int = 40
    cross_attn_norm: bool = True
    qk_norm: str = "rms_norm_across_heads"
    eps: float = 1e-6
    added_kv_proj_dim: int | None = None
    rope_dim: tuple[int, int, int] = (44, 42, 42)
    rope_theta: float = 10000.0
    guidance_cross_attn: bool = True
    zero_history_timestep: bool = True
    has_multi_term_memory_patch: bool = True
    is_amplify_history: bool = False
    history_scale_mode: str = "per_head"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.out_channels = self.out_channels or self.in_channels
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels
        if not self.cross_attn_norm:
            raise ValueError("Helios currently requires cross_attn_norm=True")
        if self.qk_norm != "rms_norm_across_heads":
            raise ValueError("Helios currently requires qk_norm='rms_norm_across_heads'")
        if self.added_kv_proj_dim is not None:
            raise ValueError("Helios added_kv_proj_dim variants are not supported")
        if not self.guidance_cross_attn:
            raise ValueError("Helios currently requires guidance_cross_attn=True")
        if not self.zero_history_timestep:
            raise ValueError("Helios currently requires zero_history_timestep=True")
        if not self.has_multi_term_memory_patch:
            raise ValueError("Helios currently requires has_multi_term_memory_patch=True")
        if self.is_amplify_history:
            raise ValueError("Helios is_amplify_history variants are not supported")
        if self.history_scale_mode != "per_head":
            raise ValueError("Helios currently requires history_scale_mode='per_head'")
        if sum(self.rope_dim) != self.attention_head_dim:
            raise ValueError(
                f"Helios rope_dim must sum to attention_head_dim, got {self.rope_dim} and {self.attention_head_dim}")
        if any(dim % 2 for dim in self.rope_dim):
            raise ValueError(f"Helios rope dimensions must be even: {self.rope_dim}")


@dataclass
class HeliosConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=HeliosArchConfig)
    prefix: str = "Helios"
