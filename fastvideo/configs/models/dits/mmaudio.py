# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig
from fastvideo.platforms import AttentionBackendEnum


class MMAudioVariantArchitecture(TypedDict):
    latent_dim: int
    hidden_dim: int
    depth: int
    fused_depth: int
    num_heads: int
    latent_seq_len: int
    v2: bool


MMAUDIO_VARIANT_ARCHITECTURES: dict[str, MMAudioVariantArchitecture] = {
    "small_16k": {
        "latent_dim": 20,
        "hidden_dim": 448,
        "depth": 12,
        "fused_depth": 8,
        "num_heads": 7,
        "latent_seq_len": 250,
        "v2": False,
    },
    "small_44k": {
        "latent_dim": 40,
        "hidden_dim": 448,
        "depth": 12,
        "fused_depth": 8,
        "num_heads": 7,
        "latent_seq_len": 345,
        "v2": False,
    },
    "medium_44k": {
        "latent_dim": 40,
        "hidden_dim": 896,
        "depth": 12,
        "fused_depth": 8,
        "num_heads": 14,
        "latent_seq_len": 345,
        "v2": False,
    },
    "large_44k": {
        "latent_dim": 40,
        "hidden_dim": 896,
        "depth": 21,
        "fused_depth": 14,
        "num_heads": 14,
        "latent_seq_len": 345,
        "v2": False,
    },
    "large_44k_v2": {
        "latent_dim": 40,
        "hidden_dim": 896,
        "depth": 21,
        "fused_depth": 14,
        "num_heads": 14,
        "latent_seq_len": 345,
        "v2": True,
    },
}

MMAUDIO_TRAINING_VARIANTS = (
    "small_16k",
    "small_44k",
    "medium_44k",
    "large_44k",
)
MMAUDIO_44K_TRAINING_VARIANTS = (
    "small_44k",
    "medium_44k",
    "large_44k",
)


def _is_mmaudio_transformer_block(name: str, module) -> bool:
    del module
    parts = name.split(".")
    return len(parts) >= 2 and parts[-1].isdigit() and parts[-2] in {"joint_blocks", "fused_blocks"}


@dataclass
class MMAudioArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=lambda: [_is_mmaudio_transformer_block])
    param_names_mapping: dict = field(default_factory=lambda: {r"^(.*)$": r"\1"})
    _supported_attention_backends: tuple[AttentionBackendEnum, ...] = (AttentionBackendEnum.TORCH_SDPA, )

    latent_dim: int = 40
    clip_dim: int = 1024
    sync_dim: int = 768
    text_dim: int = 1024
    hidden_dim: int = 896
    depth: int = 21
    fused_depth: int = 14
    num_heads: int = 14
    mlp_ratio: float = 4.0
    latent_seq_len: int = 345
    clip_seq_len: int = 64
    sync_seq_len: int = 192
    text_seq_len: int = 77
    v2: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        self.hidden_size = self.hidden_dim
        self.num_attention_heads = self.num_heads
        self.num_channels_latents = self.latent_dim
        self.in_channels = self.latent_dim
        self.out_channels = self.latent_dim


@dataclass
class MMAudioTransformerConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=MMAudioArchConfig)
    prefix: str = "MMAudio"


def get_mmaudio_transformer_config(variant: str) -> MMAudioTransformerConfig:
    """Build the exact official architecture config for ``variant``."""
    try:
        architecture = MMAUDIO_VARIANT_ARCHITECTURES[variant]
    except KeyError as exc:
        supported = ", ".join(MMAUDIO_VARIANT_ARCHITECTURES)
        raise ValueError(f"Unknown MMAudio variant {variant!r}; expected one of: {supported}") from exc
    return MMAudioTransformerConfig(arch_config=MMAudioArchConfig(**architecture), )
