import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable, Dict, Optional, Tuple, cast

import torch

from fastvideo.v1.configs.models import (DiTConfig, EncoderConfig, ModelConfig,
                                         VAEConfig)
from fastvideo.v1.configs.models.encoders import BaseEncoderOutput
from fastvideo.v1.logger import init_logger
from fastvideo.v1.utils import shallow_asdict

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
    use_cpu_offload: bool = False
    disable_autocast: bool = False

    # Model configuration
    precision: str = "bf16"

    # VAE configuration
    vae_precision: str = "fp16"
    vae_tiling: bool = True
    vae_sp: bool = True
    vae_config: VAEConfig = field(default_factory=VAEConfig)

    # DiT configuration
    dit_config: DiTConfig = field(default_factory=DiTConfig)

    # Text encoder configuration
    text_encoder_precisions: Tuple[str, ...] = field(
        default_factory=lambda: ("fp16", ))
    text_encoder_configs: Tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (EncoderConfig(), ))
    preprocess_text_funcs: Tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (preprocess_text, ))
    postprocess_text_funcs: Tuple[Callable[[BaseEncoderOutput], torch.tensor],
                                  ...] = field(default_factory=lambda:
                                               (postprocess_text, ))

    # STA (Spatial-Temporal Attention) parameters
    mask_strategy_file_path: Optional[str] = None

    # Compilation
    enable_torch_compile: bool = False

    @classmethod
    def from_pretrained(cls, model_path: str) -> "PipelineConfig":
        from fastvideo.v1.configs.pipelines.registry import (
            get_pipeline_config_cls_for_name)
        pipeline_config_cls = get_pipeline_config_cls_for_name(model_path)
        if pipeline_config_cls is not None:
            pipeline_config = pipeline_config_cls()
        else:
            logger.warning(
                "Couldn't find an optimal sampling param for %s. Using the default sampling param.",
                model_path)
            pipeline_config = cls()

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
