# SPDX-License-Identifier: Apache-2.0
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from typing import Any, cast

import torch

from fastvideo.configs.models import (DiTConfig, EncoderConfig, ModelConfig, VAEConfig, UpsamplerConfig)
from fastvideo.configs.models.encoders import BaseEncoderOutput
from fastvideo.logger import init_logger
from fastvideo.utils import shallow_asdict

logger = init_logger(__name__)


def preprocess_text(prompt: str) -> str:
    return prompt


def postprocess_text(output: BaseEncoderOutput) -> torch.tensor:
    raise NotImplementedError


# config for a single pipeline
@dataclass
class PipelineConfig:
    """Base configuration for all pipeline architectures."""
    model_path: str = ""
    pipeline_config_path: str | None = None

    # Video generation parameters
    embedded_cfg_scale: float = 6.0
    flow_shift: float | None = None
    flow_shift_sr: float | None = None
    disable_autocast: bool = False
    # When True, the scheduler's Euler update runs in fp32 outside the autocast
    # block (Diffusers-style; avoids BF16 drift over multiple steps). Flux2 sets
    # this True for reference parity; other models keep the legacy in-autocast
    # behavior to preserve existing SSIM references.
    scheduler_step_in_fp32: bool = False
    is_causal: bool = False

    # Model configuration
    dit_config: DiTConfig = field(default_factory=DiTConfig)
    dit_precision: str = "bf16"
    upsampler_config: UpsamplerConfig = field(default_factory=UpsamplerConfig)
    upsampler_precision: str = "fp32"

    # VAE configuration
    vae_config: VAEConfig = field(default_factory=VAEConfig)
    vae_precision: str = "fp32"
    # Optional decode-only precision override. When None, the decode stage falls
    # back to `vae_precision`. This lets a pipeline run a faster, lossless bf16
    # decode while keeping a higher-precision *encode* (the image/video VAE
    # encode seeds the denoising trajectory, so lowering its precision can shift
    # the output for I2V/causal models — decode is output-only and safe to lower).
    vae_decode_precision: str | None = None
    vae_tiling: bool = True
    vae_sp: bool = True

    # Image encoder configuration
    image_encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    image_encoder_precision: str = "fp32"
    # Optional multi-encoder contract. Existing pipelines continue to use the
    # singular fields above; V2A and other multimodal pipelines can opt into
    # indexed ``image_encoder``, ``image_encoder_2``, ... components.
    image_encoder_configs: tuple[EncoderConfig, ...] | None = None
    image_encoder_precisions: tuple[str, ...] | None = None

    # Text encoder configuration
    DEFAULT_TEXT_ENCODER_PRECISIONS = ("fp32", )
    text_encoder_configs: tuple[EncoderConfig, ...] = field(default_factory=lambda: (EncoderConfig(), ))
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("fp32", ))
    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(default_factory=lambda: (preprocess_text, ))
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.tensor],
                                  ...] = field(default_factory=lambda: (postprocess_text, ))

    # DMD parameters
    dmd_denoising_steps: list[int] | None = field(default=None)

    # Wan2.2 task modifiers
    ti2v_task: bool = False
    lucy_edit_task: bool = False
    boundary_ratio: float | None = None

    # Compilation
    # enable_torch_compile: bool = False

    def __setattr__(self, name: str, value: Any) -> None:
        if self.__dict__.get("_frozen") and name in self.__dataclass_fields__:
            raise AttributeError(f"{type(self).__name__}.{name} was decided by config resolution and is read-only; "
                                 f"set the typed path of pipeline_config.{name} in the input config, or decide it in a "
                                 "resolution step")
        super().__setattr__(name, value)

    def freeze(self) -> None:
        """Make the fields read-only. Nested component configs stay writable for the checkpoint data that loaders
        add to them."""
        object.__setattr__(self, "_frozen", True)

    @classmethod
    def from_pretrained(cls, model_path: str) -> "PipelineConfig":
        """
        use the pipeline class setting from model_path to match the pipeline config
        """
        from fastvideo.registry import get_pipeline_config_cls_from_name
        pipeline_config_cls = get_pipeline_config_cls_from_name(model_path)

        return cast(PipelineConfig, pipeline_config_cls(model_path=model_path))

    @classmethod
    def from_source(cls,
                    model_path: str,
                    source: "str | PipelineConfig | dict[str, Any] | None" = None) -> "PipelineConfig":
        """Build the registry ``PipelineConfig`` of ``model_path`` and update it from ``source``.

        ``source`` is a JSON file path (loaded and recorded as ``pipeline_config_path``), a mapping of field values, or
        a ``PipelineConfig`` that replaces the registry instance. The base ``PipelineConfig`` stands in when the
        registry has no class for ``model_path``. An empty path or mapping changes nothing. The result's ``model_path``
        is ``model_path``.
        """
        from fastvideo.registry import get_pipeline_config_cls_from_name

        pipeline_config_cls = get_pipeline_config_cls_from_name(model_path)
        if pipeline_config_cls is None:
            logger.warning("Couldn't find pipeline config for %s. Using the default pipeline config.", model_path)
            pipeline_config = cls()
        else:
            pipeline_config = pipeline_config_cls()

        if isinstance(source, str) and source:
            pipeline_config.load_from_json(source)
            pipeline_config.pipeline_config_path = source
        elif isinstance(source, PipelineConfig):
            pipeline_config = source
        elif isinstance(source, dict) and source:
            pipeline_config.update_pipeline_config(source)
        pipeline_config.model_path = model_path
        return pipeline_config

    def check_pipeline_config(self) -> None:
        if self.vae_sp and not self.vae_tiling:
            raise ValueError("vae_sp requires vae_tiling; set pipeline.vae_tiling to true or pipeline.vae_sp to false.")

        if len(self.text_encoder_configs) != len(self.text_encoder_precisions):
            raise ValueError(
                f"Length of text encoder configs ({len(self.text_encoder_configs)}) must be equal to length of text encoder precisions ({len(self.text_encoder_precisions)})"
            )

        if len(self.text_encoder_configs) != len(self.preprocess_text_funcs):
            raise ValueError(
                f"Length of text encoder configs ({len(self.text_encoder_configs)}) must be equal to length of text preprocessing functions ({len(self.preprocess_text_funcs)})"
            )

        if len(self.preprocess_text_funcs) != len(self.postprocess_text_funcs):
            raise ValueError(
                f"Length of text postprocess functions ({len(self.postprocess_text_funcs)}) must be equal to length of text preprocessing functions ({len(self.preprocess_text_funcs)})"
            )

    def dump_to_json(self, file_path: str):
        output_dict = shallow_asdict(self)
        del_keys = []
        for key, value in output_dict.items():
            if isinstance(value, ModelConfig):
                model_dict = asdict(value)
                # Model Arch Config should be hidden away from the users
                model_dict.pop("arch_config")
                output_dict[key] = model_dict
            elif isinstance(value, tuple) and all(isinstance(v, ModelConfig) for v in value):
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

    def update_pipeline_config(self, source_pipeline_dict: dict[str, Any]) -> None:
        for f in fields(self):
            key = f.name
            if key in source_pipeline_dict:
                current_value = getattr(self, key)
                new_value = source_pipeline_dict[key]

                # If it's a nested ModelConfig, update it recursively
                if isinstance(current_value, ModelConfig):
                    current_value.update_model_config(new_value)
                elif isinstance(current_value, tuple) and all(isinstance(v, ModelConfig) for v in current_value):
                    assert len(current_value) == len(
                        new_value), "Users shouldn't delete or add text encoder config objects in your json"
                    for target_config, source_config in zip(current_value, new_value, strict=True):
                        target_config.update_model_config(source_config)
                else:
                    setattr(self, key, new_value)

        if hasattr(self, "__post_init__"):
            self.__post_init__()
