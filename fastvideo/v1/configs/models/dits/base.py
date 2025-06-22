# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from fastvideo.v1.configs.models.base import ArchConfig, ModelConfig
from fastvideo.v1.layers.quantization import QuantizationConfig
from fastvideo.v1.platforms import AttentionBackendEnum


@dataclass
class DiTArchConfig(ArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=list)
    _compile_conditions: list = field(default_factory=list)
    _param_names_mapping: dict = field(default_factory=dict)
    _reverse_param_names_mapping: dict = field(default_factory=dict)
    _lora_param_names_mapping: dict = field(default_factory=dict)
    _supported_attention_backends: Tuple[AttentionBackendEnum, ...] = (
        AttentionBackendEnum.SLIDING_TILE_ATTN, AttentionBackendEnum.SAGE_ATTN,
        AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TORCH_SDPA,
        AttentionBackendEnum.VIDEO_SPARSE_ATTN)

    hidden_size: int = 0
    num_attention_heads: int = 0
    num_channels_latents: int = 0
    exclude_lora_layers: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self._compile_conditions:
            self._compile_conditions = self._fsdp_shard_conditions.copy()


@dataclass
class DiTConfig(ModelConfig):
    arch_config: DiTArchConfig = field(default_factory=DiTArchConfig)

    # FastVideoDiT-specific parameters
    prefix: str = ""
    quant_config: Optional[QuantizationConfig] = None

    @staticmethod
    def add_cli_args(parser: Any, prefix: str = "dit-config") -> Any:
        """Add CLI arguments for DiTConfig fields"""
        parser.add_argument(
            f"--{prefix}.prefix",
            type=str,
            dest=f"{prefix.replace('-', '_')}.prefix",
            default=DiTConfig.prefix,
            help="Prefix for the DiT model",
        )

        parser.add_argument(
            f"--{prefix}.quant-config",
            type=str,
            dest=f"{prefix.replace('-', '_')}.quant_config",
            default=None,
            help="Quantization configuration for the DiT model",
        )

        return parser
