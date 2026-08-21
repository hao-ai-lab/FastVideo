# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models.encoders.base import (
    TextEncoderArchConfig,
    TextEncoderConfig,
)


@dataclass
class LingBotWorld2UMT5ArchConfig(TextEncoderArchConfig):
    architectures: list[str] = field(default_factory=lambda: ["LingBotWorld2T5EncoderModel"])
    vocab_size: int = 256384
    dim: int = 4096
    dim_attn: int = 4096
    dim_ffn: int = 10240
    num_heads: int = 64
    num_layers: int = 24
    num_buckets: int = 32
    text_len: int = 512
    hidden_size: int = 4096
    dropout: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()
        self.tokenizer_kwargs = {
            "padding": "max_length",
            "truncation": True,
            "max_length": self.text_len,
            "add_special_tokens": True,
            "return_attention_mask": True,
            "return_tensors": "pt",
        }


@dataclass
class LingBotWorld2UMT5Config(TextEncoderConfig):
    arch_config: TextEncoderArchConfig = field(default_factory=LingBotWorld2UMT5ArchConfig)

    prefix: str = "text_encoder"
