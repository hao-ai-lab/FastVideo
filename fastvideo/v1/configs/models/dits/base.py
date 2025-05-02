from dataclasses import dataclass, field
from typing import Optional, Tuple

from fastvideo.v1.configs.models.base import ArchConfig, ModelConfig
from fastvideo.v1.configs.quantization import QuantizationConfig
from fastvideo.v1.platforms import _Backend


@dataclass
class DiTArchConfig(ArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=list)
    _param_names_mapping: dict = field(default_factory=dict)
    _supported_attention_backends: Tuple[_Backend,
                                         ...] = (_Backend.SLIDING_TILE_ATTN,
                                                 _Backend.SAGE_ATTN,
                                                 _Backend.FLASH_ATTN,
                                                 _Backend.TORCH_SDPA)

    hidden_size: int = 0
    num_attention_heads: int = 0
    num_channels_latents: int = 0


@dataclass
class DiTConfig(ModelConfig):
    arch_config: DiTArchConfig = field(default_factory=DiTArchConfig)

    # FastVideoDiT-specific parameters
    prefix: str = ""
    quant_config: Optional[QuantizationConfig] = None
