# SPDX-License-Identifier: Apache-2.0
"""Configuration for the MAGI-2 Qwen3.5-27B prompt encoder."""

from dataclasses import dataclass, field

from fastvideo.configs.models.encoders.base import TextEncoderArchConfig, TextEncoderConfig


@dataclass
class Magi2Qwen35ArchConfig(TextEncoderArchConfig):
    """Select Qwen3.5 hidden state -3 and the official token limit."""

    architectures: list[str] = field(default_factory=lambda: ["Magi2Qwen35TextEncoder"])
    vocab_size: int = 248320
    hidden_size: int = 5120
    intermediate_size: int = 17408
    num_hidden_layers: int = 64
    num_attention_heads: int = 24
    num_key_value_heads: int = 4
    head_dim: int = 256
    hidden_act: str = "silu"
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    attention_dropout: float = 0.0
    attn_output_gate: bool = True
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    full_attention_interval: int = 4
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25
    mrope_section: tuple[int, int, int] = (11, 11, 10)
    pad_token_id: int | None = None
    eos_token_id: int = 248044
    text_len: int = 7000
    hidden_state_skip_layer: int = 2
    output_hidden_states: bool = True
    tokenizer_kwargs: dict = field(default_factory=lambda: {"padding_side": "right"})
    _fsdp_shard_conditions: list = field(default_factory=list)


@dataclass
class Magi2Qwen35Config(TextEncoderConfig):
    """Load the published Qwen3.5 checkpoint through the FastVideo loader."""

    arch_config: TextEncoderArchConfig = field(default_factory=Magi2Qwen35ArchConfig)
    prefix: str = "magi2_qwen35"
