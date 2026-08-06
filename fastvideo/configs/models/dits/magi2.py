# SPDX-License-Identifier: Apache-2.0
"""Architecture configurations for the MAGI-2 preview and refiner DiTs."""

from __future__ import annotations

from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig
from fastvideo.platforms import AttentionBackendEnum


def _is_transformer_layer(name: str, module) -> bool:
    """Select a complete MAGI-2 transformer layer as an FSDP shard unit."""
    del module
    parts = name.split(".")
    return len(parts) >= 3 and parts[0] == "block" and parts[1] == "layers" and parts[2].isdigit()


@dataclass
class Magi2AttentionGatingConfig:
    """Control the learned per-head attention output gate."""

    enable: bool = True


@dataclass
class Magi2AttentionSinksConfig:
    """Configure the learned Flash Attention sink logits."""

    enable: bool = True
    sink_token_num: int = 1


@dataclass
class Magi2MHCConfig:
    """Configure multi-stream hyper-connections (MHC)."""

    enable: bool = True
    num_stream: int = 4
    alpha_init: float = 0.01


@dataclass
class Magi2MoEConfig:
    """Configure the preview model's multi-head mixture of experts."""

    num_experts: int = 256
    top_k: int = 6
    score_func: str = "sigmoid"
    route_norm: bool = True
    route_scale: float = 4.9
    moe_layers: list[int] = field(default_factory=lambda: list(range(2, 38)))
    expert_intermediate_size: int = 1280
    shared_expert_intermediate_size: int = 1280
    modality_specific_expert_intermediate_size: int = 1280
    num_heads: int = 12
    split_merge_intermediate_size: int | None = None


@dataclass
class Magi2PreviewArchConfig(DiTArchConfig):
    """Define the published 114B MAGI-2 preview transformer architecture."""

    _fsdp_shard_conditions: list = field(default_factory=lambda: [_is_transformer_layer])
    _supported_attention_backends: tuple[AttentionBackendEnum, ...] = (AttentionBackendEnum.FLASH_ATTN, )
    param_names_mapping: dict = field(default_factory=dict)
    reverse_param_names_mapping: dict = field(default_factory=dict)
    lora_param_names_mapping: dict = field(default_factory=dict)

    num_layers: int = 40
    hidden_size: int = 3072
    head_dim: int = 128
    num_query_groups: int = 24
    video_in_channels: int = 48
    audio_in_channels: int = 64
    text_in_channels: int = 5120
    params_dtype: str = "bfloat16"
    intermediate_factor: int = 4
    mm_layers: list[int] = field(default_factory=lambda: [0, 1, 38, 39])
    layer_activation_types: dict[int, str] = field(default_factory=dict)
    activation_type: str = "swiglu7"
    attn_softcap: float = -1.0
    attn_gating: Magi2AttentionGatingConfig = field(default_factory=Magi2AttentionGatingConfig)
    attn_sinks: Magi2AttentionSinksConfig = field(default_factory=Magi2AttentionSinksConfig)
    mhc_config: Magi2MHCConfig = field(default_factory=Magi2MHCConfig)
    moe_config: Magi2MoEConfig = field(default_factory=Magi2MoEConfig)

    num_heads_q: int = 0
    num_heads_kv: int = 0
    num_attention_heads: int = 0
    num_channels_latents: int = 48
    in_channels: int = 48
    out_channels: int = 48

    def __post_init__(self) -> None:
        """Derive attention dimensions and normalize nested checkpoint dictionaries."""
        super().__post_init__()
        if isinstance(self.attn_gating, dict):
            self.attn_gating = Magi2AttentionGatingConfig(**self.attn_gating)
        if isinstance(self.attn_sinks, dict):
            self.attn_sinks = Magi2AttentionSinksConfig(**self.attn_sinks)
        if isinstance(self.mhc_config, dict):
            self.mhc_config = Magi2MHCConfig(**self.mhc_config)
        if isinstance(self.moe_config, dict):
            self.moe_config = Magi2MoEConfig(**self.moe_config)
        self.num_heads_q = self.hidden_size // self.head_dim
        self.num_heads_kv = self.num_query_groups
        self.num_attention_heads = self.num_heads_q


@dataclass
class Magi2RefinerArchConfig(DiTArchConfig):
    """Define the published MAGI-2 1080p refiner transformer architecture."""

    _fsdp_shard_conditions: list = field(default_factory=lambda: [_is_transformer_layer])
    _supported_attention_backends: tuple[AttentionBackendEnum, ...] = (AttentionBackendEnum.FLASH_ATTN, )
    param_names_mapping: dict = field(default_factory=dict)
    reverse_param_names_mapping: dict = field(default_factory=dict)
    lora_param_names_mapping: dict = field(default_factory=dict)

    num_layers: int = 30
    hidden_size: int = 4096
    head_dim: int = 128
    num_query_groups: int = 8
    video_in_channels: int = 48
    audio_in_channels: int = 64
    text_in_channels: int = 5120
    checkpoint_qk_layernorm_rope: bool = False
    params_dtype: str = "bfloat16"
    mm_layers: list[int] = field(default_factory=lambda: [0, 1, 28, 29])
    local_attn_layers: list[int] = field(default_factory=lambda: list(range(30)))
    enable_attn_gating: bool = True
    activation_type: str = "swiglu7"
    layer_activation_types: dict[int, str] = field(default_factory=dict)
    post_norm_layers: list[int] = field(default_factory=list)

    num_heads_q: int = 0
    num_heads_kv: int = 0
    num_attention_heads: int = 0
    num_channels_latents: int = 48
    in_channels: int = 48
    out_channels: int = 48

    def __post_init__(self) -> None:
        """Derive the published grouped-query attention dimensions."""
        super().__post_init__()
        self.num_heads_q = self.hidden_size // self.head_dim
        self.num_heads_kv = self.num_query_groups
        self.num_attention_heads = self.num_heads_q


@dataclass
class Magi2PreviewVideoConfig(DiTConfig):
    """Load the MAGI-2 preview transformer through FastVideo's DiT loader."""

    arch_config: DiTArchConfig = field(default_factory=Magi2PreviewArchConfig)
    prefix: str = "magi2_preview"


@dataclass
class Magi2RefinerVideoConfig(DiTConfig):
    """Load the MAGI-2 refiner transformer through FastVideo's DiT loader."""

    arch_config: DiTArchConfig = field(default_factory=Magi2RefinerArchConfig)
    prefix: str = "magi2_refiner"
