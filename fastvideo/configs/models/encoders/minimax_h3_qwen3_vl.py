# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL conditioner configuration for MiniMax H3."""

from dataclasses import dataclass, field
from typing import Any

from fastvideo.configs.models.encoders.base import TextEncoderArchConfig, TextEncoderConfig


def _is_language_transformer_layer(name: str, module: Any) -> bool:
    parts = name.split(".")
    return len(parts) >= 2 and parts[-2] == "layers" and parts[-1].isdigit()


def _is_vision_transformer_layer(name: str, module: Any) -> bool:
    parts = name.split(".")
    return len(parts) >= 2 and parts[-2] == "blocks" and parts[-1].isdigit()


def _is_embeddings(name: str, module: Any) -> bool:
    return name.endswith(("embed_tokens", "patch_embed"))


def _is_vision_merger(name: str, module: Any) -> bool:
    parts = name.split(".")
    return name.endswith("visual.merger") or (len(parts) >= 2 and parts[-2] == "deepstack_merger_list"
                                              and parts[-1].isdigit())


def _is_final_norm(name: str, module: Any) -> bool:
    return name == "norm" or name.endswith("language_model.norm")


_VISION_CONFIG_MAPPING = {
    "depth": "vision_depth",
    "hidden_size": "vision_hidden_size",
    "hidden_act": "vision_hidden_act",
    "intermediate_size": "vision_intermediate_size",
    "num_heads": "vision_num_heads",
    "in_channels": "vision_in_channels",
    "patch_size": "vision_patch_size",
    "spatial_merge_size": "vision_spatial_merge_size",
    "temporal_patch_size": "vision_temporal_patch_size",
    "out_hidden_size": "vision_out_hidden_size",
    "num_position_embeddings": "vision_num_position_embeddings",
    "initializer_range": "vision_initializer_range",
    "deepstack_visual_indexes": "vision_deepstack_visual_indexes",
}

_OFFICIAL_ARCHITECTURES = {
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLModel",
}


@dataclass
class MiniMaxH3Qwen3VLArchConfig(TextEncoderArchConfig):
    """Architecture of the Qwen3-VL encoder used by MiniMax H3."""

    architectures: list[str] = field(default_factory=lambda: ["MiniMaxH3Qwen3VLConditioner"])
    vocab_size: int = 151936
    hidden_size: int = 5120
    intermediate_size: int = 25600
    num_hidden_layers: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    # H3 text conditioning truncates at 1,024 tokens. The parquet loader reads
    # the same field when padding stored embeddings and attention masks.
    text_len: int = 1024
    hidden_act: str = "silu"
    max_position_embeddings: int = 262144
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    attention_bias: bool = False
    attention_dropout: float = 0.0
    rope_theta: float = 5000000.0
    rope_scaling: dict[str, Any] | None = field(default_factory=lambda: {
        "mrope_interleaved": True,
        "mrope_section": [24, 20, 20],
        "rope_type": "default",
    })
    mrope_interleaved: bool = True
    mrope_section: tuple[int, int, int] = (24, 20, 20)

    bos_token_id: int = 151643
    eos_token_id: int = 151645
    pad_token_id: int | None = None
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    image_token_id: int = 151655
    video_token_id: int = 151656
    tie_word_embeddings: bool = False

    vision_depth: int = 27
    vision_hidden_size: int = 1152
    vision_hidden_act: str = "gelu_pytorch_tanh"
    vision_intermediate_size: int = 4304
    vision_num_heads: int = 16
    vision_in_channels: int = 3
    vision_patch_size: int = 16
    vision_spatial_merge_size: int = 2
    vision_temporal_patch_size: int = 2
    vision_out_hidden_size: int = 5120
    vision_num_position_embeddings: int = 2304
    vision_initializer_range: float = 0.02
    vision_deepstack_visual_indexes: tuple[int, ...] = (8, 16, 24)

    output_hidden_states: bool = True
    stacked_params_mapping: list[tuple[str, str, str | int]] = field(default_factory=list)
    _fsdp_shard_conditions: list = field(default_factory=lambda: [
        _is_language_transformer_layer,
        _is_vision_transformer_layer,
        _is_embeddings,
        _is_vision_merger,
        _is_final_norm,
    ])

    def __post_init__(self) -> None:
        rope_scaling = dict(self.rope_scaling or {})
        self.mrope_interleaved = bool(rope_scaling.get("mrope_interleaved", self.mrope_interleaved))
        if not self.mrope_interleaved:
            raise ValueError("MiniMax H3 Qwen3-VL only supports interleaved multimodal RoPE.")
        section = rope_scaling.get("mrope_section", self.mrope_section)
        if not isinstance(section, list | tuple) or len(section) != 3:
            raise ValueError("MiniMax H3 Qwen3-VL mRoPE requires exactly three section sizes.")
        self.mrope_section = (int(section[0]), int(section[1]), int(section[2]))
        if sum(self.mrope_section) * 2 != self.head_dim:
            raise ValueError("MiniMax H3 Qwen3-VL mRoPE sections must cover exactly half of each attention head.")
        rope_scaling["mrope_interleaved"] = self.mrope_interleaved
        rope_scaling["mrope_section"] = list(self.mrope_section)
        rope_scaling.setdefault("rope_type", "default")
        self.rope_scaling = rope_scaling

        if self.vision_out_hidden_size != self.hidden_size:
            raise ValueError("MiniMax H3 Qwen3-VL vision_out_hidden_size must match the language hidden_size "
                             "for visual-token and DeepStack injection.")

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
        flattened = dict(source_model_dict)

        text_config = flattened.pop("text_config", {})
        flattened.update(dict(text_config))

        vision_config = flattened.pop("vision_config", {})
        for source_name, target_name in _VISION_CONFIG_MAPPING.items():
            if source_name in vision_config:
                flattened[target_name] = vision_config[source_name]

        architectures = flattened.get("architectures")
        if isinstance(architectures, str):
            architectures = [architectures]
        if isinstance(architectures, list | tuple) and any(architecture in _OFFICIAL_ARCHITECTURES
                                                           for architecture in architectures):
            flattened["architectures"] = ["MiniMaxH3Qwen3VLConditioner"]

        section = flattened.get("mrope_section")
        if isinstance(section, list):
            flattened["mrope_section"] = tuple(section)
        deepstack_indexes = flattened.get("vision_deepstack_visual_indexes")
        if isinstance(deepstack_indexes, list):
            flattened["vision_deepstack_visual_indexes"] = tuple(deepstack_indexes)

        super().update_model_arch(flattened)
