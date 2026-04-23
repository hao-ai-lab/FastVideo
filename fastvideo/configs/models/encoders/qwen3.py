# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field, fields
from typing import Any

from fastvideo.configs.models.encoders.base import TextEncoderArchConfig, TextEncoderConfig


def _is_transformer_layer(n: str, m) -> bool:
    return "layers" in n and str.isdigit(n.split(".")[-1])


def _is_embeddings(n: str, m) -> bool:
    return n.endswith("embed_tokens")


def _is_final_norm(n: str, m) -> bool:
    return n.endswith("norm")


@dataclass
class Qwen3ArchConfig(TextEncoderArchConfig):
    architectures: list[str] = field(default_factory=lambda: ["Qwen3Model"])
    model_type: str = "qwen3"

    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_theta: float = 1000000.0
    rope_scaling: dict[str, Any] | None = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    mlp_bias: bool = False

    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = None

    hidden_state_skip_layer: int = 2
    text_len: int = 512
    output_hidden_states: bool = True
    stacked_params_mapping: list[tuple[str, str, str | int]] = field(default_factory=lambda: [
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
    ])
    _fsdp_shard_conditions: list = field(default_factory=lambda: [_is_transformer_layer, _is_embeddings, _is_final_norm])


@dataclass
class Qwen3Config(TextEncoderConfig):
    arch_config: TextEncoderArchConfig = field(default_factory=Qwen3ArchConfig)
    prefix: str = "qwen3"
    is_chat_model: bool = False

    def update_model_arch(self, source_model_dict: dict[str, Any]) -> None:
        """Lenient arch update for HF Qwen3 configs.

        Qwen3 checkpoints may carry auxiliary config keys that are not required
        by FastVideo runtime. We only copy known arch fields.
        """
        arch_config = self.arch_config
        valid_fields = {f.name for f in fields(arch_config)}

        for key, value in source_model_dict.items():
            if key in valid_fields:
                setattr(arch_config, key, value)

        if hasattr(arch_config, "__post_init__"):
            arch_config.__post_init__()
