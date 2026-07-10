# SPDX-License-Identifier: Apache-2.0
"""Mistral3 text encoder configuration for full Flux2."""
from dataclasses import dataclass, field

from fastvideo.configs.models.encoders.base import (
    TextEncoderArchConfig,
    TextEncoderConfig,
)


@dataclass
class Mistral3TextArchConfig(TextEncoderArchConfig):
    """Architecture config for the Mistral3 text encoder used by full Flux2."""

    architectures: list[str] = field(default_factory=lambda: ["Mistral3ForConditionalGeneration"])
    hidden_size: int = 5120
    num_hidden_layers: int = 40
    text_len: int = 512
    output_hidden_states: bool = True
    # Mistral3 (full Flux2) ships a multimodal processor; load via AutoProcessor.
    require_processor: bool = True

    def __post_init__(self) -> None:
        self.tokenizer_kwargs = {
            "padding": "max_length",
            "truncation": True,
            "max_length": self.text_len,
            "return_tensors": "pt",
        }


@dataclass
class Mistral3TextConfig(TextEncoderConfig):
    """Top-level config for the Mistral3 full Flux2 text encoder."""

    arch_config: TextEncoderArchConfig = field(default_factory=Mistral3TextArchConfig)
    prefix: str = "mistral3"
    is_chat_model: bool = True
