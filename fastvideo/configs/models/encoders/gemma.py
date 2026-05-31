# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.configs.models.encoders.base import (
    TextEncoderArchConfig,
    TextEncoderConfig,
)


def _is_feature_extractor_linear(n: str, m) -> bool:
    # LTX-2.3 (caption_proj_before_connector) introduces separate
    # video/audio feature extractor linears; keep the LTX-2.0 name too.
    return (n.endswith("feature_extractor_linear") or n.endswith("video_feature_extractor_linear")
            or n.endswith("audio_feature_extractor_linear"))


def _is_embeddings(n: str, m) -> bool:
    return n.endswith("embeddings_connector") or n.endswith("audio_embeddings_connector")


def _is_gemma_model(n: str, m) -> bool:
    return "_gemma_model" in n


@dataclass
class LTX2GemmaArchConfig(TextEncoderArchConfig):
    architectures: list[str] = field(default_factory=lambda: ["LTX2GemmaTextEncoderModel"])
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
    # LTX-2.3 text-stack connector fields (default OFF == LTX-2.0 behavior).
    video_feature_extractor_out_features: int | None = None
    audio_feature_extractor_out_features: int | None = None
    caption_proj_before_connector: bool = False
    caption_projection_first_linear: bool = True
    caption_proj_input_norm: bool = True
    caption_projection_second_linear: bool = True

    connector_num_attention_heads: int = 30
    connector_attention_head_dim: int = 128
    connector_num_layers: int = 2
    # Separate audio connector geometry (None falls back to the video values).
    audio_connector_num_attention_heads: int | None = None
    audio_connector_attention_head_dim: int | None = None
    audio_connector_num_layers: int | None = None
    connector_positional_embedding_theta: float = 10000.0
    connector_positional_embedding_max_pos: list[int] = field(default_factory=lambda: [4096])
    connector_rope_type: str = "split"
    connector_double_precision_rope: bool = False
    connector_apply_gated_attention: bool = False
    connector_num_learnable_registers: int | None = 128

    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [_is_feature_extractor_linear, _is_embeddings, _is_gemma_model])

    def __post_init__(self) -> None:
        super().__post_init__()
        self.tokenizer_kwargs["padding"] = "max_length"


@dataclass
class LTX2GemmaConfig(TextEncoderConfig):
    arch_config: TextEncoderArchConfig = field(default_factory=LTX2GemmaArchConfig)

    prefix: str = "ltx2_gemma"
