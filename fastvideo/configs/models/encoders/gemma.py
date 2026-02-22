# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models.encoders.base import (
    TextEncoderArchConfig,
    TextEncoderConfig,
)


def _is_feature_extractor_linear(n: str, m) -> bool:
    return n.endswith("feature_extractor_linear")


def _is_embeddings(n: str, m) -> bool:
    return n.endswith("embeddings_connector") or n.endswith(
        "audio_embeddings_connector")


def _is_gemma_model(n: str, m) -> bool:
    return "_gemma_model" in n


@dataclass
class LTX2GemmaArchConfig(TextEncoderArchConfig):
    architectures: list[str] = field(
        default_factory=lambda: ["LTX2GemmaTextEncoderModel"])
    hidden_size: int = 3840
    num_hidden_layers: int = 48
    num_attention_heads: int = 30
    text_len: int = 1024
    pad_token_id: int = 0
    eos_token_id: int = 2

    gemma_model_path: str = ""
    gemma_dtype: str = "bfloat16"
    padding_side: str = "left"

    feature_extractor_in_features: int = 3840 * 49
    feature_extractor_out_features: int = 3840

    connector_num_attention_heads: int = 30
    connector_attention_head_dim: int = 128
    connector_num_layers: int = 2
    connector_positional_embedding_theta: float = 10000.0
    connector_positional_embedding_max_pos: list[int] = field(
        default_factory=lambda: [4096])
    connector_rope_type: str = "split"
    connector_double_precision_rope: bool = False
    connector_num_learnable_registers: int | None = 128

    _fsdp_shard_conditions: list = field(
        default_factory=lambda:
        [_is_feature_extractor_linear, _is_embeddings, _is_gemma_model])

    def __post_init__(self) -> None:
        super().__post_init__()
        self.tokenizer_kwargs["padding"] = "max_length"


@dataclass
class LTX2GemmaConfig(TextEncoderConfig):
    arch_config: TextEncoderArchConfig = field(
        default_factory=LTX2GemmaArchConfig)

    prefix: str = "ltx2_gemma"
