from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from fastvideo.v1.configs.models.base import ArchConfig, ModelConfig
from fastvideo.v1.configs.quantization import QuantizationConfig
from fastvideo.v1.platforms import _Backend


@dataclass
class EncoderArchConfig(ArchConfig):
    architectures: List[str] = field(default_factory=lambda: [])
    _supported_attention_backends: Tuple[_Backend, ...] = (_Backend.FLASH_ATTN,
                                                           _Backend.TORCH_SDPA)
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


@dataclass
class ImageEncoderArchConfig(EncoderArchConfig):
    pass


@dataclass
class EncoderConfig(ModelConfig):
    arch_config: ArchConfig = EncoderArchConfig()

    prefix: str = ""
    quant_config: Optional[QuantizationConfig] = None
    lora_config: Optional[Any] = None


@dataclass
class TextEncoderConfig(EncoderConfig):
    arch_config: ArchConfig = TextEncoderArchConfig()


@dataclass
class ImageEncoderConfig(EncoderConfig):
    arch_config: ArchConfig = ImageEncoderArchConfig()
