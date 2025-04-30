import json
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional

from fastvideo.v1.configs.models import (DiTConfig, EncoderConfig, ModelConfig,
                                         VAEConfig)
from fastvideo.v1.logger import init_logger
from fastvideo.v1.utils import shallow_asdict

logger = init_logger(__name__)


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
    vae_config: VAEConfig = VAEConfig()

    # DiT configuration
    dit_config: DiTConfig = DiTConfig()

    # Text encoder configuration
    text_encoder_precision: str = "fp16"
    text_encoder_config: EncoderConfig = EncoderConfig()

    # STA (Spatial-Temporal Attention) parameters
    mask_strategy_file_path: Optional[str] = None
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

        return pipeline_config

    def dump_to_json(self, file_path: str):
        output_dict = shallow_asdict(self)
        for key, value in output_dict.items():
            if isinstance(value, ModelConfig):
                model_dict = asdict(value)
                # Model Arch Config should be hidden away from the users
                model_dict.pop("arch_config")
                output_dict[key] = model_dict

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
