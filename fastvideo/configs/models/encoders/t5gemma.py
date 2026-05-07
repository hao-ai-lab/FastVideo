# SPDX-License-Identifier: Apache-2.0
"""Config for the T5-Gemma encoder used by daVinci-MagiHuman.

The reference pipeline uses `transformers.models.t5gemma.T5GemmaEncoderModel`
on `google/t5gemma-9b-9b-ul2`. That is a gated Google repository, so the
encoder weights are not bundled inside GAIR/daVinci-MagiHuman; they are
loaded from the T5-Gemma HF repo directly.

Encoder shape (verified from google/t5gemma-9b-9b-ul2/config.json):
    layers=42, hidden=3584, heads=16, kv_heads=8, head_dim=256,
    intermediate=14336, rope_theta=10000.0, max_pos=8192,
    layer_types alternate sliding_attention / full_attention.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from fastvideo.configs.models.encoders.base import (
    TextEncoderArchConfig,
    TextEncoderConfig,
)


def _is_t5gemma_model(n: str, m) -> bool:
    return n.endswith("t5gemma_model") or n.endswith("_t5gemma_model")


@dataclass
class T5GemmaEncoderArchConfig(TextEncoderArchConfig):
    architectures: list[str] = field(default_factory=lambda: ["T5GemmaEncoderModel"])

    hidden_size: int = 3584
    num_hidden_layers: int = 42
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 256
    intermediate_size: int = 14336
    max_position_embeddings: int = 8192
    rope_theta: float = 10000.0
    vocab_size: int = 256000

    # MagiHuman fixes prompt embed length at 640 via pad_or_trim.
    text_len: int = 640

    pad_token_id: int = 0
    eos_token_id: int = 1

    # Path to the upstream gated repo. When set, the FastVideo loader will
    # pull the encoder directly via `T5GemmaEncoderModel.from_pretrained`.
    t5gemma_model_path: str = "google/t5gemma-9b-9b-ul2"
    t5gemma_dtype: str = "bfloat16"

    _fsdp_shard_conditions: list = field(default_factory=lambda: [_is_t5gemma_model])

    def __post_init__(self) -> None:
        super().__post_init__()
        # WHY: upstream `t5_gemma_model.py:25` tokenizes without
        # padding/max_length, then `prompt_process.py` pad_or_trim-s the
        # encoded states. Keep only tensor return here so
        # MagiHumanLatentPreparationStage can pad/trim post-encode while
        # preserving the real original prompt length.
        self.tokenizer_kwargs.pop("truncation", None)
        self.tokenizer_kwargs.pop("max_length", None)
        self.tokenizer_kwargs.pop("padding", None)


@dataclass
class T5GemmaEncoderConfig(TextEncoderConfig):
    arch_config: TextEncoderArchConfig = field(default_factory=T5GemmaEncoderArchConfig)

    prefix: str = "t5gemma"
