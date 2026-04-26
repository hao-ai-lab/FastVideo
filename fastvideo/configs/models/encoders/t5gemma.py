# SPDX-License-Identifier: Apache-2.0
"""T5Gemma-9B encoder config for daVinci-MagiHuman.

google/t5gemma-9b is an encoder-decoder; we use the encoder only.
hidden_size (d_model) = 3584, matching daVinci text_in_channels.
text_len = 640 as used in daVinci official inference.
"""
from dataclasses import dataclass, field

from fastvideo.configs.models.encoders.t5 import T5ArchConfig, T5Config


@dataclass
class T5GemmaArchConfig(T5ArchConfig):
    """Architecture config for google/t5gemma-9b encoder."""
    # Gemma-9B hidden dim
    d_model: int = 3584
    # Gemma-style attention: 16 heads, head dim 256
    num_heads: int = 16
    d_kv: int = 256
    # Gemma 9B has 42 encoder layers
    num_layers: int = 42
    # d_ff: 4× hidden = 14336 (Gemma MLP)
    d_ff: int = 14336
    feed_forward_proj: str = "gated-gelu"
    # daVinci uses 640-token text sequences
    text_len: int = 640
    # Vocab from SentencePiece tokenizer shared with T5
    vocab_size: int = 32128


@dataclass
class T5GemmaConfig(T5Config):
    """Top-level encoder config for google/t5gemma-9b."""
    arch_config: T5GemmaArchConfig = field(
        default_factory=T5GemmaArchConfig)
    prefix: str = "t5gemma"
