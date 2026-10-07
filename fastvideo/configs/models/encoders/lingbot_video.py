# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL text-only encoder configuration used by LingBot-Video."""

from dataclasses import dataclass, field

from fastvideo.configs.models.encoders.base import TextEncoderArchConfig
from fastvideo.configs.models.encoders.qwen3 import Qwen3TextArchConfig, Qwen3TextConfig


@dataclass
class LingBotVideoQwen3VLTextArchConfig(Qwen3TextArchConfig):
    """Exact Qwen3-VL language-model architecture released with LingBot-Video."""

    architectures: list[str] = field(default_factory=lambda: ["LingBotVideoQwen3VLTextModel"])
    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    rope_scaling: dict | None = None
    mrope_interleaved: bool = True
    mrope_section: tuple[int, int, int] = (24, 20, 20)
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    pad_token_id: int = 151643
    text_len: int = 37698
    output_hidden_states: bool = True
    require_processor: bool = True

    def __post_init__(self) -> None:
        """Match the official processor call used by LingBotVideoPipeline."""
        self.tokenizer_kwargs = {
            "truncation": True,
            "max_length": self.text_len,
            "padding": "longest",
            "return_tensors": "pt",
        }


@dataclass
class LingBotVideoQwen3VLTextConfig(Qwen3TextConfig):
    """FastVideo loader config for the LingBot-Video text-only Qwen3-VL path."""

    arch_config: TextEncoderArchConfig = field(default_factory=LingBotVideoQwen3VLTextArchConfig)
    prefix: str = "language_model"
    is_chat_model: bool = False
