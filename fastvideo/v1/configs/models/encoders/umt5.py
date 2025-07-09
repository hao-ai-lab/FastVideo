# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.v1.configs.models.encoders.base import (TextEncoderArchConfig,
                                                       TextEncoderConfig)


def _is_transformer_layer(n: str, m) -> bool:
    return "block" in n and str.isdigit(n.split(".")[-1])


def _is_embeddings(n: str, m) -> bool:
    return n.endswith("shared")


def _is_final_layernorm(n: str, m) -> bool:
    return n.endswith("final_layer_norm")


@dataclass
class UMT5ArchConfig(TextEncoderArchConfig):
    vocab_size: int = 32128
    d_model: int = 1024
    d_kv: int = 128
    d_ff: int = 65536
    num_layers: int = 24
    num_decoder_layers: int | None = 24
    num_heads: int = 128
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    initializer_factor: float = 1.0
    feed_forward_proj: str = "relu"
    dense_act_fn: str = ""
    is_gated_act: bool = False
    is_encoder_decoder: bool = True
    use_cache: bool = True
    tokenizer_class: str = "T5Tokenizer"
    tie_word_embeddings: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 1
    decoder_start_token_id: int = 0
    classifier_dropout: float = 0.0
    n_positions: int = 512
    task_specific_params: dict | None = None
    text_len: int = 512
    stacked_params_mapping: list[tuple[str, str,
                                       str]] = field(default_factory=lambda: [
                                           # (param_name, shard_name, shard_id)
                                           (".qkv_proj", ".q", "q"),
                                           (".qkv_proj", ".k", "k"),
                                           (".qkv_proj", ".v", "v"),
                                       ])
    _fsdp_shard_conditions: list = field(
        default_factory=lambda:
        [_is_transformer_layer, _is_embeddings, _is_final_layernorm])

    # Referenced from https://github.com/huggingface/transformers/blob/main/src/transformers/models/umt5/configuration_umt5.py
    def __post_init__(self):
        super().__post_init__()
        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"
        
        if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {self.feed_forward_proj} is not a valid activation function of the dense layer. "
                "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                "'gated-gelu' or 'relu'"
            )
        
        if self.feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"

        self.tokenizer_kwargs = {
            "padding": "max_length",
            "truncation": True,
            "max_length": self.text_len,
            "add_special_tokens": True,
            "return_attention_mask": True,
            "return_tensors": "pt",
        }


@dataclass
class UMT5Config(TextEncoderConfig):
    arch_config: TextEncoderArchConfig = field(default_factory=UMT5ArchConfig)

    prefix: str = "umt5"