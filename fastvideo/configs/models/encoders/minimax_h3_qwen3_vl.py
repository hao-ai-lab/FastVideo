# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL conditioner configuration for MiniMax H3."""

from dataclasses import dataclass, field
from typing import Any

from fastvideo.configs.models.encoders.base import TextEncoderArchConfig, TextEncoderConfig


@dataclass
class MiniMaxH3Qwen3VLArchConfig(TextEncoderArchConfig):
    """Architecture of the Qwen3-VL encoder used by MiniMax H3."""

    architectures: list[str] = field(default_factory=lambda: ["MiniMaxH3Qwen3VLConditioner"])
    hidden_size: int = 5120
    num_hidden_layers: int = 64
    output_hidden_states: bool = True

    def __post_init__(self) -> None:
        # H3 builds its presentation in the pipeline stage and tokenizes each
        # segment verbatim without adding tokenizer-owned presentation tokens.
        self.tokenizer_kwargs = {"add_special_tokens": False}


@dataclass
class MiniMaxH3Qwen3VLConfig(TextEncoderConfig):
    """FastVideo loader config for H3's base Qwen3-VL encoder."""

    arch_config: TextEncoderArchConfig = field(default_factory=MiniMaxH3Qwen3VLArchConfig)
    prefix: str = "minimax_h3_qwen3_vl"
    is_chat_model: bool = False

    def update_model_arch(self, source_model_dict: dict[str, Any]) -> None:
        source_model_dict = dict(source_model_dict)
        if source_model_dict.get("architectures") in (
            ["Qwen3VLForConditionalGeneration"],
            ["Qwen3VLModel"],
        ):
            source_model_dict["architectures"] = ["MiniMaxH3Qwen3VLConditioner"]
        super().update_model_arch(source_model_dict)
