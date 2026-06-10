# SPDX-License-Identifier: Apache-2.0
# Ported from SGLang: python/sglang/multimodal_gen/configs/models/encoders/qwen3.py
"""Qwen3 text encoder configurations for FastVideo diffusion models.

Two variants coexist:
- ``Qwen3ArchConfig`` / ``Qwen3Config``: defaults from Ovis2.5-2B (Qwen3-2B),
  used by Ovis-Image (chat-template path with a sliced system prompt).
- ``Qwen3TextArchConfig`` / ``Qwen3TextConfig``: used by Flux2 Klein.
"""
from dataclasses import dataclass, field
from typing import Any

from fastvideo.configs.models.encoders.base import (TextEncoderArchConfig, TextEncoderConfig)


def _is_transformer_layer(n: str, m: Any) -> bool:
    return "layers" in n and str.isdigit(n.split(".")[-1])


def _is_embeddings(n: str, m: Any) -> bool:
    return n.endswith("embed_tokens")


def _is_final_norm(n: str, m: Any) -> bool:
    return n.endswith("norm")


@dataclass
class Qwen3ArchConfig(TextEncoderArchConfig):
    """Qwen3 text encoder arch config; defaults from Ovis2.5-2B (Qwen3-2B)."""

    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 40960
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-06
    use_cache: bool = True
    tie_word_embeddings: bool = True
    rope_theta: float = 1000000.0
    rope_scaling: dict | None = None
    use_sliding_window: bool = False
    sliding_window: int | None = None
    max_window_layers: int = 28
    attention_dropout: float = 0.0
    attention_bias: bool = False
    head_dim: int = 128

    bos_token_id: int = 151643
    eos_token_id: int = 151645
    dtype: str = "float32"
    _attn_implementation_autoset: bool = True
    layer_types: list[str] = field(default_factory=lambda: ["full_attention"] * 28)

    hidden_state_skip_layer: int = 0
    text_len: int = 256
    # Ovis-Image prepends a system prompt; its tokens are sliced off downstream.
    user_prompt_begin_id: int = 28

    # (param_name, shard_name, shard_id) for fusing q/k/v and gate/up.
    stacked_params_mapping: list[tuple[str, str, str]] = field(default_factory=lambda: [
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".gate_up_proj", ".gate_proj", 0),  # type: ignore
        (".gate_up_proj", ".up_proj", 1),  # type: ignore
    ])

    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [_is_transformer_layer, _is_embeddings, _is_final_norm])

    def __post_init__(self):
        super().__post_init__()
        # Chat-template tokenization; max_length leaves room for the 28 system tokens.
        self.tokenizer_kwargs = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "padding": "max_length",
            "max_length": self.text_len + self.user_prompt_begin_id,
            "truncation": True,
            "return_tensors": "pt",
            "enable_thinking": False,
        }


@dataclass
class Qwen3TextArchConfig(TextEncoderArchConfig):
    """Architecture config for Qwen3 text encoder.

    Qwen3 is similar to LLaMA but with QK-Norm (RMSNorm on Q and K before attention).
    Used by Flux2 Klein.
    """

    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 40960
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int = 151643
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    tie_word_embeddings: bool = True
    rope_theta: float = 1000000.0
    rope_scaling: dict | None = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    mlp_bias: bool = False
    head_dim: int = 128
    text_len: int = 512
    output_hidden_states: bool = True  # Klein needs hidden states from layers 9, 18, 27

    stacked_params_mapping: list[tuple[str, str, str | int]] = field(default_factory=lambda: [
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
    ])
    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [_is_transformer_layer, _is_embeddings, _is_final_norm])

    def __post_init__(self) -> None:
        self.tokenizer_kwargs = {
            "padding": "max_length",
            "truncation": True,
            "max_length": self.text_len,
            "return_tensors": "pt",
        }


@dataclass
class Qwen3Config(TextEncoderConfig):
    """Configuration for Qwen3 text encoder (Ovis-Image)."""

    arch_config: TextEncoderArchConfig = field(default_factory=Qwen3ArchConfig)
    prefix: str = "qwen3"
    is_chat_model: bool = True


@dataclass
class Qwen3TextConfig(TextEncoderConfig):
    """Top-level config for Qwen3 text encoder (Flux2 Klein)."""

    arch_config: TextEncoderArchConfig = field(default_factory=Qwen3TextArchConfig)
    prefix: str = "qwen3"
    is_chat_model: bool = True
