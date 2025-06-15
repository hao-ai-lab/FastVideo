# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from fastvideo.v1.configs.models.base import ArchConfig, ModelConfig
from fastvideo.v1.layers.quantization import QuantizationConfig
from fastvideo.v1.platforms import AttentionBackendEnum


@dataclass
class EncoderArchConfig(ArchConfig):
    architectures: List[str] = field(default_factory=lambda: [])
    _supported_attention_backends: Tuple[AttentionBackendEnum, ...] = (
        AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TORCH_SDPA)
    output_hidden_states: bool = False
    use_return_dict: bool = True


@dataclass
class TextEncoderArchConfig(EncoderArchConfig):
    vocab_size: int = 0
    hidden_size: int = 0
    num_hidden_layers: int = 0
    num_attention_heads: int = 0
    pad_token_id: int = 0
    eos_token_id: int = 0
    text_len: int = 0
    hidden_state_skip_layer: int = 0
    decoder_start_token_id: int = 0
    output_past: bool = True
    scalable_attention: bool = True
    tie_word_embeddings: bool = False

    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tokenizer_kwargs = {
            "truncation": True,
            "max_length": self.text_len,
            "return_tensors": "pt",
        }


@dataclass
class ImageEncoderArchConfig(EncoderArchConfig):
    pass


@dataclass
class BaseEncoderOutput:
    last_hidden_state: Optional[torch.FloatTensor] = None
    pooler_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    attention_mask: Optional[torch.Tensor] = None


@dataclass
class EncoderConfig(ModelConfig):
    arch_config: ArchConfig = field(default_factory=EncoderArchConfig)

    prefix: str = ""
    quant_config: Optional[QuantizationConfig] = None
    lora_config: Optional[Any] = None


@dataclass
class TextEncoderConfig(EncoderConfig):
    arch_config: ArchConfig = field(default_factory=TextEncoderArchConfig)


@dataclass
class ImageEncoderConfig(EncoderConfig):
    arch_config: ArchConfig = field(default_factory=ImageEncoderArchConfig)
