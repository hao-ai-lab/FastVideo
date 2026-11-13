# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL architecture used for Qwen-Image-2.1 conditioning."""

from dataclasses import dataclass, field
from typing import Any

from fastvideo.configs.models.base import ModelConfig
from fastvideo.configs.models.encoders.minimax_h3_qwen3_vl import (
    MiniMaxH3Qwen3VLArchConfig,
    MiniMaxH3Qwen3VLConfig,
    _OFFICIAL_ARCHITECTURES,
    _VISION_CONFIG_MAPPING,
)


@dataclass
class QwenImage21Qwen3VLArchConfig(MiniMaxH3Qwen3VLArchConfig):
    architectures: list[str] = field(default_factory=lambda: ["QwenImage21Qwen3VLConditioner"])
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    output_hidden_state_index: int = 36
    num_hidden_layers_override: int | None = None
    num_attention_heads: int = 32
    vision_out_hidden_size: int = 4096
    # The image pipeline's processor owns prompt lengths, including reference
    # image tokens; the H3 text-only truncation limit is inappropriate here.
    text_len: int = 262144
    require_processor: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.output_hidden_state_index != self.num_hidden_layers:
            raise ValueError("Qwen-Image-2.1 conditioning requires the last decoder layer before final normalization")
        if self.num_hidden_layers_override is not None:
            raise ValueError("Qwen-Image-2.1 conditioning requires all text decoder layers")
        if self.hidden_act != "silu" or self.vision_hidden_act != "gelu_pytorch_tanh":
            raise ValueError("Qwen-Image-2.1 requires SiLU text MLPs and tanh-approximate GELU vision MLPs")
        if (self.rope_scaling or {}).get("rope_type", "default") != "default":
            raise ValueError("Qwen-Image-2.1 conditioning supports default multimodal RoPE")
        self.tokenizer_kwargs = {
            "padding": True,
            "padding_side": "left",
            "truncation": False,
            "return_tensors": "pt",
        }


@dataclass
class QwenImage21Qwen3VLConfig(MiniMaxH3Qwen3VLConfig):
    arch_config: QwenImage21Qwen3VLArchConfig = field(default_factory=QwenImage21Qwen3VLArchConfig)
    prefix: str = "qwen_image21_qwen3_vl"

    def update_model_arch(self, source_model_dict: dict[str, Any]) -> None:
        flattened = dict(source_model_dict)
        flattened.update(dict(flattened.pop("text_config", {})))
        vision = flattened.pop("vision_config", {})
        for source_name, target_name in _VISION_CONFIG_MAPPING.items():
            if source_name in vision:
                flattened[target_name] = vision[source_name]

        architectures = flattened.get("architectures", [])
        if isinstance(architectures, str):
            architectures = [architectures]
        if any(architecture in _OFFICIAL_ARCHITECTURES for architecture in architectures):
            flattened["architectures"] = ["QwenImage21Qwen3VLConditioner"]
        if "num_hidden_layers" in flattened:
            flattened["output_hidden_state_index"] = flattened["num_hidden_layers"]
        for name in ("mrope_section", "vision_deepstack_visual_indexes"):
            if isinstance(flattened.get(name), list):
                flattened[name] = tuple(flattened[name])
        # Transformers 5 serializes the same default RoPE parameters under a
        # renamed field. Keep the checkpoint's frequencies in either format.
        if "rope_parameters" in flattened:
            parameters = dict(flattened["rope_parameters"])
            flattened["rope_scaling"] = parameters
            if "rope_theta" in parameters:
                flattened["rope_theta"] = parameters["rope_theta"]
        ModelConfig.update_model_arch(self, flattened)
