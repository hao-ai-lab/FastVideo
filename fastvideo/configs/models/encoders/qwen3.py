# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models.encoders.base import (TextEncoderArchConfig,
                                                    TextEncoderConfig)


def _is_transformer_layer(n: str, m) -> bool:
    return "layers" in n and str.isdigit(n.split(".")[-1])


def _is_embeddings(n: str, m) -> bool:
    return n.endswith("embed_tokens")


def _is_final_norm(n: str, m) -> bool:
    return n.endswith("norm")


@dataclass
class Qwen3ArchConfig(TextEncoderArchConfig):
    """Architecture config for Qwen3 text encoder (used in Ovis-Image)."""

    # Model architecture - defaults from Ovis2.5-2B (Qwen3-2B)
    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 6144  # Actual value from Ovis2.5-2B
    num_hidden_layers: int = 28  # Actual value from Ovis2.5-2B
    num_attention_heads: int = 16
    num_key_value_heads: int = 8  # Actual value from Ovis2.5-2B
    hidden_act: str = "silu"
    max_position_embeddings: int = 40960  # Actual value from Ovis2.5-2B
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-06
    use_cache: bool = True
    tie_word_embeddings: bool = True
    rope_theta: float = 1000000.0
    rope_scaling: dict | None = None
    use_sliding_window: bool = False
    sliding_window: int | None = None  # Can be None
    max_window_layers: int = 28  # Actual value from Ovis2.5-2B
    attention_dropout: float = 0.0
    attention_bias: bool = False
    head_dim: int = 128

    # HuggingFace transformers fields
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    dtype: str = "float32"
    _attn_implementation_autoset: bool = True
    layer_types: list[str] = field(
        default_factory=lambda: ["full_attention"] * 28)

    # FastVideo-specific settings
    hidden_state_skip_layer: int = 0
    text_len: int = 256

    # Ovis-Image uses system prompt tokens (28 tokens) prepended to user tokens
    user_prompt_begin_id: int = 28

    # Qwen3-specific stacked params
    stacked_params_mapping: list[tuple[str, str, str]] = field(
        default_factory=lambda: [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),  # type: ignore
            (".gate_up_proj", ".up_proj", 1),  # type: ignore
        ])

    _fsdp_shard_conditions: list = field(
        default_factory=lambda:
        [_is_transformer_layer, _is_embeddings, _is_final_norm])

    def __post_init__(self):
        super().__post_init__()
        # Override tokenizer_kwargs for apply_chat_template
        # Ovis-Image uses chat template with system prompt (28 tokens prepended)
        # Total max_length = text_len + user_prompt_begin_id
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
class Qwen3Config(TextEncoderConfig):
    """Configuration for Qwen3 text encoder."""

    arch_config: TextEncoderArchConfig = field(default_factory=Qwen3ArchConfig)
    prefix: str = "qwen3"
    is_chat_model: bool = True
