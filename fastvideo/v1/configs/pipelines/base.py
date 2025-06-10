# SPDX-License-Identifier: Apache-2.0
import argparse
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable, Dict, Optional, Tuple, cast

import torch

from fastvideo.v1.configs.models import (DiTConfig, EncoderConfig, ModelConfig,
                                         VAEConfig)
from fastvideo.v1.configs.models.encoders import BaseEncoderOutput
from fastvideo.v1.configs.utils import update_config_from_args
from fastvideo.v1.logger import init_logger
from fastvideo.v1.utils import FlexibleArgumentParser, StoreBoolean, shallow_asdict

logger = init_logger(__name__)


def preprocess_text(prompt: str) -> str:
    return prompt


def postprocess_text(output: BaseEncoderOutput) -> torch.tensor:
    raise NotImplementedError


@dataclass
class PipelineConfig:
    """Base configuration for all pipeline architectures."""
    # Video generation parameters
    embedded_cfg_scale: float = 6.0
    flow_shift: Optional[float] = None
    disable_autocast: bool = False

    # Model configuration
    dit_config: DiTConfig = field(default_factory=DiTConfig)
    dit_precision: str = "bf16"

    # VAE configuration
    vae_config: VAEConfig = field(default_factory=VAEConfig)
    vae_precision: str = "fp16"
    vae_tiling: bool = True
    vae_sp: bool = True

    # Image encoder configuration
    image_encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    image_encoder_precision: str = "fp32"

    # Text encoder configuration
    DEFAULT_TEXT_ENCODER_PRECISIONS = ("fp16",)
    text_encoder_configs: Tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (EncoderConfig(), ))
    text_encoder_precisions: Tuple[str, ...] = field(
        default_factory=lambda: ("fp16", ))
    preprocess_text_funcs: Tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (preprocess_text, ))
    postprocess_text_funcs: Tuple[Callable[[BaseEncoderOutput], torch.tensor],
                                  ...] = field(default_factory=lambda:
                                               (postprocess_text, ))

    # StepVideo specific parameters
    pos_magic: Optional[str] = None
    neg_magic: Optional[str] = None
    timesteps_scale: Optional[bool] = None

    # STA (Spatial-Temporal Attention) parameters
    mask_strategy_file_path: Optional[str] = None
    STA_mode: str = "STA_inference"
    skip_time_steps: int = 15

    # Compilation
    enable_torch_compile: bool = False

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser, prefix: str = "pipeline-config") -> FlexibleArgumentParser:
        # 
        parser.add_argument(
            f"--{prefix}.embedded-cfg-scale",
            type=float,
            dest=f"{prefix.replace('-', '_')}.embedded_cfg_scale",
            default=PipelineConfig.embedded_cfg_scale,
            help="Embedded CFG scale",
        )
        parser.add_argument(
            f"--{prefix}.flow-shift",
            type=float,
            dest=f"{prefix.replace('-', '_')}.flow_shift",
            default=PipelineConfig.flow_shift,
            help="Flow shift parameter",
        )

        # DiT configuration
        parser.add_argument(
            f"--{prefix}.dit-precision",
            type=str,
            dest=f"{prefix.replace('-', '_')}.dit_precision",
            default=PipelineConfig.dit_precision,
            choices=["fp32", "fp16", "bf16"],
            help="Precision for the DiT model",
        )

        # VAE configuration
        parser.add_argument(
            f"--{prefix}.vae-precision",
            type=str,
            dest=f"{prefix.replace('-', '_')}.vae_precision",
            default=PipelineConfig.vae_precision,
            choices=["fp32", "fp16", "bf16"],
            help="Precision for VAE",
        )
        parser.add_argument(
            f"--{prefix}.vae-tiling",
            action=StoreBoolean,
            dest=f"{prefix.replace('-', '_')}.vae_tiling",
            default=PipelineConfig.vae_tiling,
            help="Enable VAE tiling",
        )
        parser.add_argument(
            f"--{prefix}.vae-sp",
            action=StoreBoolean,
            dest=f"{prefix.replace('-', '_')}.vae_sp",
            help="Enable VAE spatial parallelism",
        )

        # Text encoder configuration
        parser.add_argument(
            f"--{prefix}.text-encoder-precisions",
            nargs="+",
            type=str,
            dest=f"{prefix.replace('-', '_')}.text_encoder_precisions",
            default=PipelineConfig.DEFAULT_TEXT_ENCODER_PRECISIONS,
            choices=["fp32", "fp16", "bf16"],
            help="Precision for each text encoder",
        )

        # Image encoder configuration
        parser.add_argument(
            f"--{prefix}.image-encoder-precision",
            type=str,
            dest=f"{prefix.replace('-', '_')}.image_encoder_precision",
            default=PipelineConfig.image_encoder_precision,
            choices=["fp32", "fp16", "bf16"],
            help="Precision for image encoder",
        )
        parser.add_argument(
            f"--{prefix}.pos_magic",
            type=str,
            dest=f"{prefix.replace('-', '_')}.pos_magic",
            default=PipelineConfig.pos_magic,
            help="Positive magic prompt for sampling, used in stepvideo",
        )
        parser.add_argument(
            f"--{prefix}.neg_magic",
            type=str,
            dest=f"{prefix.replace('-', '_')}.neg_magic",
            default=PipelineConfig.neg_magic,
            help="Negative magic prompt for sampling, used in stepvideo",
        )
        parser.add_argument(
            f"--{prefix}.timesteps_scale",
            type=bool,
            dest=f"{prefix.replace('-', '_')}.timesteps_scale",
            default=PipelineConfig.timesteps_scale,
            help="Bool for applying scheduler scale in set_timesteps, used in stepvideo",
        )

        # Add VAE configuration arguments
        from fastvideo.v1.configs.models.vaes.base import VAEConfig
        VAEConfig.add_cli_args(parser, prefix=f"{prefix}.vae-config")

        # Add DiT configuration arguments
        from fastvideo.v1.configs.models.dits.base import DiTConfig
        DiTConfig.add_cli_args(parser, prefix=f"{prefix}.dit-config")
    
        return parser

    def update_config_from_cli_args(self, args: argparse.Namespace) -> None:
        update_config_from_args(self, args, "pipeline_config")
        update_config_from_args(self.vae_config, args, "pipeline_config.vae_config")
        update_config_from_args(self.dit_config, args, "pipeline_config.dit_config")
        # attrs = [attr.name for attr in dataclasses.fields(cls)]
        # kwargs = {}
        # for attr in attrs:
        #     if attr == 'vae_config':
        #         kwargs[attr] = VAEConfig.from_cli_args(args)
        #     elif attr == 'dit_config':
        #         kwargs[attr] = DiTConfig.from_cli_args(args)
        #     else:
        #         kwargs[attr] = getattr(args, attr, getattr(cls, attr, None))
        # return cls(**kwargs)


    @classmethod
    def from_pretrained(cls, model_path: str) -> "PipelineConfig":
        from fastvideo.v1.configs.pipelines.registry import (
            get_pipeline_config_from_name)
        pipeline_config = get_pipeline_config_from_name(model_path)

        return cast(PipelineConfig, pipeline_config)

    def dump_to_json(self, file_path: str):
        output_dict = shallow_asdict(self)
        del_keys = []
        for key, value in output_dict.items():
            if isinstance(value, ModelConfig):
                model_dict = asdict(value)
                # Model Arch Config should be hidden away from the users
                model_dict.pop("arch_config")
                output_dict[key] = model_dict
            elif isinstance(value, tuple) and all(
                    isinstance(v, ModelConfig) for v in value):
                model_dicts = []
                for v in value:
                    model_dict = asdict(v)
                    # Model Arch Config should be hidden away from the users
                    model_dict.pop("arch_config")
                    model_dicts.append(model_dict)
                output_dict[key] = model_dicts
            elif isinstance(value, tuple) and all(callable(f) for f in value):
                # Skip dumping functions
                del_keys.append(key)

        for key in del_keys:
            output_dict.pop(key, None)

        with open(file_path, "w") as f:
            json.dump(output_dict, f, indent=2)

    def load_from_json(self, file_path: str):
        with open(file_path) as f:
            input_pipeline_dict = json.load(f)
        self.update_pipeline_config(input_pipeline_dict)

    def update_pipeline_config(self, source_pipeline_dict: Dict[str,
                                                                Any]) -> None:
        for f in fields(self):
            key = f.name
            if key in source_pipeline_dict:
                current_value = getattr(self, key)
                new_value = source_pipeline_dict[key]

                # If it's a nested ModelConfig, update it recursively
                if isinstance(current_value, ModelConfig):
                    current_value.update_model_config(new_value)
                elif isinstance(current_value, tuple) and all(
                        isinstance(v, ModelConfig) for v in current_value):
                    assert len(current_value) == len(
                        new_value
                    ), "Users shouldn't delete or add text encoder config objects in your json"
                    for target_config, source_config in zip(
                            current_value, new_value):
                        target_config.update_model_config(source_config)
                else:
                    setattr(self, key, new_value)

        if hasattr(self, "__post_init__"):
            self.__post_init__()


@dataclass
class SlidingTileAttnConfig(PipelineConfig):
    """Configuration for sliding tile attention."""

    # Override any BaseConfig defaults as needed
    # Add sliding tile specific parameters
    window_size: int = 16
    stride: int = 8

    # You can provide custom defaults for inherited fields
    height: int = 576
    width: int = 1024

    # Additional configuration specific to sliding tile attention
    pad_to_square: bool = False
    use_overlap_optimization: bool = True
