# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL encoder configurations used by LingBot-Video."""

from dataclasses import dataclass, field

from fastvideo.configs.models.encoders.base import TextEncoderArchConfig
from fastvideo.configs.models.encoders.qwen3 import Qwen3TextArchConfig, Qwen3TextConfig


@dataclass
class LingBotVideoQwen3VLTextArchConfig(Qwen3TextArchConfig):
    """Exact Qwen3-VL language-model architecture released with LingBot-Video."""

    architectures: list[str] = field(default_factory=lambda: ["LingBotVideoQwen3VLTextModel"])
    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    rope_scaling: dict | None = None
    mrope_interleaved: bool = True
    mrope_section: tuple[int, int, int] = (24, 20, 20)
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    pad_token_id: int = 151643
    text_len: int = 37698
    output_hidden_states: bool = True
    require_processor: bool = True

    def __post_init__(self) -> None:
        """Match the official processor call used by LingBotVideoPipeline."""
        self.tokenizer_kwargs = {
            "truncation": True,
            "max_length": self.text_len,
            "padding": "longest",
            "return_tensors": "pt",
        }


@dataclass
class LingBotVideoQwen3VLVisionArchConfig:
    """Vision-tower architecture released with LingBot-Video's Qwen3-VL."""

    depth: int = 24
    hidden_act: str = "gelu_pytorch_tanh"
    hidden_size: int = 1024
    in_channels: int = 3
    intermediate_size: int = 4096
    num_heads: int = 16
    num_position_embeddings: int = 2304
    out_hidden_size: int = 2560
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    deepstack_visual_indexes: tuple[int, ...] = (5, 11, 17)


@dataclass
class LingBotVideoQwen3VLArchConfig(LingBotVideoQwen3VLTextArchConfig):
    """Compound language-and-vision architecture needed by TI2V conditioning."""

    architectures: list[str] = field(default_factory=lambda: ["LingBotVideoQwen3VLModel"])
    vision_config: dict = field(default_factory=lambda: LingBotVideoQwen3VLVisionArchConfig().__dict__.copy())
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653


@dataclass
class LingBotVideoQwen3VLTextConfig(Qwen3TextConfig):
    """FastVideo loader config for the LingBot-Video text-only Qwen3-VL path."""

    arch_config: TextEncoderArchConfig = field(default_factory=LingBotVideoQwen3VLTextArchConfig)
    prefix: str = "language_model"
    is_chat_model: bool = False


@dataclass
class LingBotVideoQwen3VLConfig(LingBotVideoQwen3VLTextConfig):
    """FastVideo loader config for LingBot-Video's compound Qwen3-VL model."""

    arch_config: TextEncoderArchConfig = field(default_factory=LingBotVideoQwen3VLArchConfig)
