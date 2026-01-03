# SPDX-License-Identifier: Apache-2.0
"""Config for Reason1 (Qwen2.5-VL) text encoder."""

from dataclasses import dataclass, field
from typing import Any

from fastvideo.configs.models.encoders.base import TextEncoderArchConfig, TextEncoderConfig


@dataclass
class Reason1ArchConfig(TextEncoderArchConfig):
    """Architecture settings (defaults match Qwen2.5-VL-7B-Instruct)."""

    architectures: list[str] = field(
        default_factory=lambda: ["Qwen2_5_VLForConditionalGeneration"])
    model_type: str = "qwen2_5_vl"

    vocab_size: int = 152064
    hidden_size: int = 3584
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    intermediate_size: int = 18944

    text_len: int = 512
    hidden_state_skip_layer: int = 0
    bos_token_id: int = 151643
    pad_token_id: int = 151643
    eos_token_id: int = 151645

    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_token_id: int = 151654
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653

    vision_config: dict[str, Any] | None = None

    rope_theta: float = 1000000.0
    rope_scaling: dict[str, Any] | None = field(default_factory=lambda: {
        "type": "mrope",
        "mrope_section": [16, 24, 24]
    })
    max_position_embeddings: int = 128000
    max_window_layers: int = 28

    embedding_concat_strategy: str = "mean_pooling"
    n_layers_per_group: int = 5
    num_embedding_padding_tokens: int = 512

    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6

    use_sliding_window: bool = False
    sliding_window: int = 32768

    tie_word_embeddings: bool = False
    use_cache: bool = False
    output_hidden_states: bool = True

    torch_dtype: str = "bfloat16"
    _attn_implementation: str = "flash_attention_2"


@dataclass
class Reason1Config(TextEncoderConfig):
    """Reason1 text encoder config."""

    arch_config: Reason1ArchConfig = field(default_factory=Reason1ArchConfig)
    tokenizer_type: str = "Qwen/Qwen2.5-VL-7B-Instruct"
