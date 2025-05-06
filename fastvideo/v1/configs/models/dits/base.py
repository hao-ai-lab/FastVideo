from dataclasses import dataclass, field
from typing import Optional, Tuple, Any, Union, Dict
import argparse
import torch
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


    # in_channels: int = 16,
    # out_channels: int = 16,
    # num_attention_heads: int = 24,
    # attention_head_dim: int = 128,
    # num_layers: int = 20,
    # num_single_layers: int = 40,
    # num_refiner_layers: int = 2,
    # mlp_ratio: float = 4.0,
    # patch_size: int = 2,
    # patch_size_t: int = 1,
    # qk_norm: str = "rms_norm",
    # guidance_embeds: bool = True,
    # text_embed_dim: int = 4096,
    # pooled_projection_dim: int = 768,
    # rope_theta: float = 256.0,
    # # rope_axes_dim: Tuple[int] = (16, 56, 56),
    # rope_axes_dim: Tuple[int] = (0, 0, 0),
    # dtype: str = "bfloat16",
    
    # patch_size: int = 2
    # patch_size_t: int = 1
    # in_channels: int = 16
    # out_channels: int = 16
    # num_attention_heads: int = 24
    # attention_head_dim: int = 128
    # mlp_ratio: float = 4.0
    # num_layers: int = 20
    # num_single_layers: int = 40
    # num_refiner_layers: int = 2
    # rope_axes_dim: Tuple[int, int, int] = (16, 56, 56)
    # guidance_embeds: bool = False
    # dtype: Optional[torch.dtype] = None
    # text_embed_dim: int = 4096
    # pooled_projection_dim: int = 768
    # rope_theta: int = 256
    # qk_norm: str = "rms_norm"


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
        
        # Add quantization config as a string
        parser.add_argument(
            f"--{prefix}.quant-config",
            type=str,
            dest=f"{prefix.replace('-', '_')}.quant_config",
            default=None,
            help="Quantization configuration for the DiT model",
        )
        
        return parser
    
    # @classmethod
    # def from_cli_args(cls, args: Union[argparse.Namespace, Dict]) -> "DiTConfig":
    #     """Create a DiTConfig from command-line arguments"""
    #     # pipeline_config = PipelineConfig.from_pretrained(args.model_path)
    #     dit_config = pipeline_config.dit_config
    #     # Extract prefix from args
    #     prefix_key = "dit_config.prefix"
    #     if prefix_key in args and args[prefix_key] is not None:
    #         dit_config.prefix = args[prefix_key]
        
    #     # Extract quant_config from args
    #     quant_config_key = "dit_config.quant_config"
    #     dit_config.quant_config = None

    #     print(f"dit_config!: {dit_config}")
        
    #     return dit_config
