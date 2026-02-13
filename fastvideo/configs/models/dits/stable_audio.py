# SPDX-License-Identifier: Apache-2.0
"""
Stable Audio DiT config for FastVideo.
"""
from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class StableAudioDiTArchConfig(DiTArchConfig):
    """Arch config for Stable Audio DiT."""

    # Iterator strips model.model. prefix; map inner keys to wrapper's model.*
    param_names_mapping: dict = field(default_factory=lambda: {
        r"^(.*)$": r"model.\1",
    })
    reverse_param_names_mapping: dict = field(default_factory=dict)
    lora_param_names_mapping: dict = field(default_factory=dict)
    _fsdp_shard_conditions: list = field(default_factory=list)

    # HF config fields (from transformer/config.json)
    attention_head_dim: int = 64
    cross_attention_dim: int = 768
    cross_attention_input_dim: int = 768
    global_states_input_dim: int = 1536
    num_key_value_attention_heads: int = 12
    num_layers: int = 24
    sample_size: int = 1024
    time_proj_dim: int = 256


@dataclass
class StableAudioDiTConfig(DiTConfig):
    """Config for Stable Audio DiffusionTransformer."""

    arch_config: DiTArchConfig = field(default_factory=StableAudioDiTArchConfig)
    unified_checkpoint_path: str | None = None
    transformer_key_prefix: str = "model.model."
