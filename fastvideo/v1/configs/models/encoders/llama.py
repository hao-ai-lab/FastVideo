# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Optional

from fastvideo.v1.configs.models.encoders.base import (TextEncoderArchConfig,
                                                       TextEncoderConfig)


@dataclass
class LlamaArchConfig(TextEncoderArchConfig):
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    pretraining_tp: int = 1
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling: Optional[float] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    mlp_bias: bool = False
    head_dim: Optional[int] = None
    hidden_state_skip_layer: int = 2
    text_len: int = 256


@dataclass
class LlamaConfig(TextEncoderConfig):
    arch_config: TextEncoderArchConfig = field(default_factory=LlamaArchConfig)

    prefix: str = "llama"
