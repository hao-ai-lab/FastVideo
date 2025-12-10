# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Optional

from fastvideo.configs.models.encoders.base import (TextEncoderArchConfig,
                                                    TextEncoderConfig)

def _is_transformer_layer(n: str, m) -> bool:
    return "layers" in n and str.isdigit(n.split(".")[-1])


def _is_embeddings(n: str, m) -> bool:
    return n.endswith("embed_tokens")


def _is_final_norm(n: str, m) -> bool:
    return n.endswith("norm")


@dataclass
class Qwen2_5_VLArchConfig(TextEncoderArchConfig):
    vocab_size: int = 152064
    hidden_size: int = 8192
    intermediate_size: int = 29568
    num_hidden_layers: int = 80
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-05
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_theta: float = 1000000.0
    use_sliding_window: bool = False
    sliding_window: int = 4096
    max_window_layers: int = 80
    layer_types: list = field(default_factory=list)
    attention_dropout: float = 0.0
    rope_scaling: Optional[dict] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    vision_token_id: int = 151654
    model_type: str = "qwen2_5_vl_text"
    dtype: str = "bfloat16"
    
    stacked_params_mapping: list[tuple[str, str, str]] = field(
        default_factory=lambda: [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ])
    _fsdp_shard_conditions: list = field(
        default_factory=lambda:
        [_is_transformer_layer, _is_embeddings, _is_final_norm])

    def __post_init__(self):
        super().__post_init__()
        self.sliding_window = self.sliding_window if self.use_sliding_window else None
        # for backward compatibility
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            if self.rope_scaling["type"] == "mrope":
                self.rope_scaling["type"] = "default"
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

@dataclass
class Qwen2_5_VLConfig(TextEncoderConfig):
    arch_config: TextEncoderArchConfig = field(default_factory=Qwen2_5_VLArchConfig)
    prefix: str = "qwen2_5_vl"