# fastvideo/v1/configs/models/encoders/bert.py
from dataclasses import dataclass, field
from typing import Optional

from fastvideo.v1.configs.models.encoders.base import (TextEncoderArchConfig,
                                                       TextEncoderConfig)

# ---------- Architecture-level hyper-parameters -----------------
@dataclass
class BertArchConfig(TextEncoderArchConfig):
    vocab_size: int = 47020
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    
    hidden_act: str = "gelu"
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    position_embedding_type: str = "absolute"
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02

    attention_probs_dropout_prob: float = 0.1
    hidden_dropout_prob: float = 0.1
    use_cache: bool = True                       # KV cache toggle
    pad_token_id: int = 0
    bos_token_id: int = 0
    eos_token_id: int = 2

    text_len: int = 77

    # === Pooler / NSP head ===================================================
    pooler_fc_size: int = 768
    pooler_num_attention_heads: int = 12
    pooler_num_fc_layers: int = 3
    pooler_size_per_head: int = 128
    pooler_type: str = "first_token_transform"
    
    classifier_dropout: Optional[float] = None
    directionality: str = "bidi"
    output_past: bool = True 

# ---------- Top-level config wrapper (YAML-friendly) ------------
@dataclass
class BertConfig(TextEncoderConfig):
    arch_config: TextEncoderArchConfig = field(default_factory=BertArchConfig)
    prefix: str = "bert"