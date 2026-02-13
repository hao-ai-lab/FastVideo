# SPDX-License-Identifier: Apache-2.0
"""
Central registry for FastVideo pipelines and model configuration discovery.

This module mirrors the organization of sglang's registry while keeping
FastVideo's legacy behavior and mappings intact.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.configs.pipelines.cosmos import CosmosConfig
from fastvideo.configs.pipelines.cosmos2_5 import Cosmos25Config
from fastvideo.configs.pipelines.hunyuan import FastHunyuanConfig, HunyuanConfig
from fastvideo.configs.pipelines.hunyuan15 import (
    Hunyuan15T2V480PConfig, Hunyuan15I2V480PStepDistilledConfig,
    Hunyuan15T2V720PConfig, Hunyuan15I2V720PConfig, Hunyuan15SR1080PConfig)
from fastvideo.configs.pipelines.hyworld import HYWorldConfig
from fastvideo.configs.pipelines.lingbotworld import LingBotWorldI2V480PConfig
from fastvideo.configs.pipelines.longcat import LongCatT2V480PConfig
from fastvideo.configs.pipelines.ltx2 import LTX2T2VConfig
from fastvideo.configs.pipelines.lucy_edit import LucyEditConfig
from fastvideo.configs.pipelines.stepvideo import StepVideoT2VConfig
from fastvideo.configs.pipelines.turbodiffusion import (
    TurboDiffusionI2V_A14B_Config,
    TurboDiffusionT2V_14B_Config,
    TurboDiffusionT2V_1_3B_Config,
)
from fastvideo.configs.pipelines.wan import (
    FastWan2_1_T2V_480P_Config,
    FastWan2_2_TI2V_5B_Config,
    MatrixGameI2V480PConfig,
    SelfForcingWan2_2_T2V480PConfig,
    SelfForcingWanT2V480PConfig,
    WANV2VConfig,
    Wan2_2_I2V_A14B_Config,
    Wan2_2_T2V_A14B_Config,
    Wan2_2_TI2V_5B_Config,
    WanI2V480PConfig,
    WanI2V720PConfig,
    WanT2V480PConfig,
    WanT2V720PConfig,
)
from fastvideo.configs.pipelines.sd35 import SD35Config
from fastvideo.configs.sample.base import SamplingParam
from fastvideo.configs.sample.cosmos import (
    Cosmos_Predict2_2B_Video2World_SamplingParam, )
from fastvideo.configs.sample.cosmos2_5 import Cosmos25SamplingParamBase
from fastvideo.configs.sample.hunyuan import (FastHunyuanSamplingParam,
                                              HunyuanSamplingParam)
from fastvideo.configs.sample.hunyuan15 import (
    Hunyuan15_480P_SamplingParam,
    Hunyuan15_480P_StepDistilled_I2V_SamplingParam,
    Hunyuan15_720P_SamplingParam, Hunyuan15_720P_Distilled_I2V_SamplingParam,
    Hunyuan15_SR_1080P_SamplingParam)
from fastvideo.configs.sample.hyworld import HYWorld_SamplingParam
from fastvideo.configs.sample.lingbotworld import LingBotWorld_SamplingParam
from fastvideo.configs.sample.ltx2 import (LTX2BaseSamplingParam,
                                           LTX2DistilledSamplingParam)
from fastvideo.configs.sample.lucy_edit import LucyEditSamplingParam
from fastvideo.configs.sample.stepvideo import StepVideoT2VSamplingParam
from fastvideo.configs.sample.turbodiffusion import (
    TurboDiffusionI2V_A14B_SamplingParam,
    TurboDiffusionT2V_14B_SamplingParam,
    TurboDiffusionT2V_1_3B_SamplingParam,
)
from fastvideo.configs.sample.wan import (
    FastWanT2V480P_SamplingParam,
    MatrixGame2_SamplingParam,
    SelfForcingWan2_1_T2V_1_3B_480P_SamplingParam,
    SelfForcingWan2_2_T2V_A14B_480P_SamplingParam,
    Wan2_1_Fun_1_3B_Control_SamplingParam,
    Wan2_1_Fun_1_3B_InP_SamplingParam,
    Wan2_2_I2V_A14B_SamplingParam,
    Wan2_2_T2V_A14B_SamplingParam,
    Wan2_2_TI2V_5B_SamplingParam,
    WanI2V_14B_480P_SamplingParam,
    WanI2V_14B_720P_SamplingParam,
    WanT2V_14B_SamplingParam,
    WanT2V_1_3B_SamplingParam,
)
from fastvideo.configs.sample.sd35 import SD35SamplingParam

from fastvideo.fastvideo_args import WorkloadType
from fastvideo.logger import init_logger
from fastvideo.utils import (maybe_download_model_index,
                             verify_model_config_and_directory)

logger = init_logger(__name__)

if TYPE_CHECKING:
    from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
    from fastvideo.pipelines.pipeline_registry import PipelineType

# --- Part 1: Pipeline Discovery ---

_PIPELINE_REGISTRY: dict[str, dict[str, type[ComposedPipelineBase]]] = {}

# Registry for pipeline configuration classes (for single-file weights without
# model_index.json). Maps pipeline_class_name -> (PipelineConfig, SamplingParam)
_PIPELINE_CONFIG_REGISTRY: dict[str, tuple[type[PipelineConfig],
                                           type[SamplingParam]]] = {}


def _discover_and_register_pipelines() -> None:
    if _PIPELINE_REGISTRY:
        return

    from fastvideo.pipelines.pipeline_registry import import_pipeline_classes

    pipeline_classes = import_pipeline_classes()
    for pipeline_type, pipeline_dict in pipeline_classes.items():
        _PIPELINE_REGISTRY[pipeline_type] = pipeline_dict
        for pipeline_cls in pipeline_dict.values():
            if pipeline_cls is None:
                continue
            if hasattr(pipeline_cls, "pipeline_config_cls") and hasattr(
                    pipeline_cls, "sampling_params_cls"):
                _PIPELINE_CONFIG_REGISTRY[pipeline_cls.__name__] = (
                    pipeline_cls.pipeline_config_cls,
                    pipeline_cls.sampling_params_cls,
                )


def get_pipeline_config_classes(
    pipeline_class_name: str
) -> tuple[type[PipelineConfig], type[SamplingParam]] | None:
    _discover_and_register_pipelines()
    return _PIPELINE_CONFIG_REGISTRY.get(pipeline_class_name)


# --- Part 2: Config Registration ---


@dataclasses.dataclass
class ConfigInfo:
    """Encapsulates sampling + pipeline config classes for a model family."""

    sampling_param_cls: type[SamplingParam] | None
    pipeline_config_cls: type[PipelineConfig]


# The central registry mapping a model name to its configuration information
_CONFIG_REGISTRY: dict[str, ConfigInfo] = {}

# Mappings from Hugging Face model paths to our internal model names
_MODEL_HF_PATH_TO_NAME: dict[str, str] = {}

# Detectors to identify model families from paths or class names
_MODEL_NAME_DETECTORS: list[tuple[str, Callable[[str], bool]]] = []


def register_configs(
    sampling_param_cls: type[SamplingParam] | None,
    pipeline_config_cls: type[PipelineConfig],
    hf_model_paths: list[str] | None = None,
    model_detectors: list[Callable[[str], bool]] | None = None,
) -> None:
    """Register config classes for a model family."""
    model_id = str(len(_CONFIG_REGISTRY))

    _CONFIG_REGISTRY[model_id] = ConfigInfo(
        sampling_param_cls=sampling_param_cls,
        pipeline_config_cls=pipeline_config_cls,
    )

    if hf_model_paths:
        for path in hf_model_paths:
            if path in _MODEL_HF_PATH_TO_NAME:
                logger.warning(
                    "Model path '%s' is already mapped to '%s' and will be overwritten by '%s'.",
                    path, _MODEL_HF_PATH_TO_NAME[path], model_id)
            _MODEL_HF_PATH_TO_NAME[path] = model_id

    if model_detectors:
        for detector in model_detectors:
            _MODEL_NAME_DETECTORS.append((model_id, detector))


def get_model_short_name(model_id: str) -> str:
    if "/" in model_id:
        return model_id.split("/")[-1]
    return model_id


def _get_config_info(
    model_path: str,
    *,
    raise_on_missing: bool = True,
) -> ConfigInfo | None:
    # 1. Exact match
    if model_path in _MODEL_HF_PATH_TO_NAME:
        model_id = _MODEL_HF_PATH_TO_NAME[model_path]
        logger.debug("Resolved model path '%s' from exact path match.",
                     model_path)
        return _CONFIG_REGISTRY.get(model_id)

    # 2. Partial match: use short model name.
    model_name = get_model_short_name(model_path.lower())
    all_model_hf_paths = sorted(_MODEL_HF_PATH_TO_NAME.keys(),
                                key=len,
                                reverse=True)
    for registered_model_hf_id in all_model_hf_paths:
        registered_model_name = get_model_short_name(
            registered_model_hf_id.lower())
        if registered_model_name == model_name:
            logger.debug("Resolved model name '%s' from partial path match.",
                         registered_model_hf_id)
            model_id = _MODEL_HF_PATH_TO_NAME[registered_model_hf_id]
            return _CONFIG_REGISTRY.get(model_id)

    # 3. Use detectors (path or model_index pipeline name).
    if os.path.exists(model_path):
        config = verify_model_config_and_directory(model_path)
    else:
        config = maybe_download_model_index(model_path)

    pipeline_name = config.get("_class_name", "").lower()

    matched_model_names: list[str] = []
    for model_id, detector in _MODEL_NAME_DETECTORS:
        if detector(model_path.lower()) or detector(pipeline_name):
            logger.debug("Matched model name '%s' using a registered detector.",
                         model_id)
            matched_model_names.append(model_id)

    if matched_model_names:
        if len(matched_model_names) > 1:
            logger.warning(
                "Multiple models matched for path '%s': %s. Using the first matched: '%s'.",
                model_path,
                matched_model_names,
                matched_model_names[0],
            )
        model_id = matched_model_names[0]
        return _CONFIG_REGISTRY.get(model_id)

    if raise_on_missing:
        raise RuntimeError(f"No model info found for model path: {model_path}")
    return None


def _register_configs() -> None:
    # LTX-2 (base)
    register_configs(
        sampling_param_cls=LTX2BaseSamplingParam,
        pipeline_config_cls=LTX2T2VConfig,
        hf_model_paths=[
            "Lightricks/LTX-2",
            "FastVideo/LTX2-base",
            "FastVideo/LTX2-Diffusers",
        ],
        model_detectors=[
            lambda path: ("ltx2" in path.lower() or "ltx-2" in path.lower()) and
            "distilled" not in path.lower(),
        ],
    )
    # LTX-2 (distilled)
    register_configs(
        sampling_param_cls=LTX2DistilledSamplingParam,
        pipeline_config_cls=LTX2T2VConfig,
        hf_model_paths=[
            "FastVideo/LTX2-Distilled-Diffusers",
        ],
        model_detectors=[
            lambda path: ("ltx2" in path.lower() or "ltx-2" in path.lower()) and
            "distilled" in path.lower(),
        ],
    )

    # Hunyuan 1.5 (specific)
    register_configs(
        sampling_param_cls=Hunyuan15_480P_SamplingParam,
        pipeline_config_cls=Hunyuan15T2V480PConfig,
        hf_model_paths=[
            "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        ],
        model_detectors=[
            lambda path: any(token in path.lower() for token in (
                "hunyuan15",
                "hunyuanvideo15",
                "hunyuanvideo-1.5",
                "hunyuanvideo_1.5",
            )),
        ],
    )
    register_configs(
        sampling_param_cls=Hunyuan15_480P_StepDistilled_I2V_SamplingParam,
        pipeline_config_cls=Hunyuan15I2V480PStepDistilledConfig,
        hf_model_paths=[
            "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled",
        ],
    )
    register_configs(
        sampling_param_cls=Hunyuan15_720P_SamplingParam,
        pipeline_config_cls=Hunyuan15T2V720PConfig,
        hf_model_paths=[
            "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
        ],
    )
    register_configs(
        sampling_param_cls=Hunyuan15_720P_Distilled_I2V_SamplingParam,
        pipeline_config_cls=Hunyuan15I2V720PConfig,
        hf_model_paths=[
            "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v_distilled",
        ],
    )
    register_configs(
        sampling_param_cls=Hunyuan15_SR_1080P_SamplingParam,
        pipeline_config_cls=Hunyuan15SR1080PConfig,
        hf_model_paths=[
            "weizhou03/HunyuanVideo-1.5-Diffusers-1080p",
            "weizhou03/HunyuanVideo-1.5-Diffusers-1080p-2SR"
        ],
    )

    # Hunyuan
    register_configs(
        sampling_param_cls=HunyuanSamplingParam,
        pipeline_config_cls=HunyuanConfig,
        hf_model_paths=[
            "hunyuanvideo-community/HunyuanVideo",
        ],
        model_detectors=[lambda path: "hunyuan" in path.lower()],
    )
    register_configs(
        sampling_param_cls=FastHunyuanSamplingParam,
        pipeline_config_cls=FastHunyuanConfig,
        hf_model_paths=[
            "FastVideo/FastHunyuan-diffusers",
        ],
    )

    # HYWorld
    register_configs(
        sampling_param_cls=HYWorld_SamplingParam,
        pipeline_config_cls=HYWorldConfig,
        hf_model_paths=[
            "FastVideo/HY-WorldPlay-Bidirectional-Diffusers",
        ],
        model_detectors=[lambda path: "hyworld" in path.lower()],
    )

    # LingBotWorld
    register_configs(
        sampling_param_cls=LingBotWorld_SamplingParam,
        pipeline_config_cls=LingBotWorldI2V480PConfig,
        hf_model_paths=[
            "FastVideo/LingBot-World-Base-Cam-Diffusers",
        ],
        model_detectors=[
            lambda path:
            ("lingbotworld" in path.lower() or "lingbot-world" in path.lower())
        ],
    )

    # LongCat
    register_configs(
        sampling_param_cls=None,
        pipeline_config_cls=LongCatT2V480PConfig,
        hf_model_paths=[
            "FastVideo/LongCat-Video-T2V-Diffusers",
            "FastVideo/LongCat-Video-I2V-Diffusers",
            "FastVideo/LongCat-Video-VC-Diffusers",
        ],
        model_detectors=[
            lambda path: "longcatimagetovideo" in path.lower(),
            lambda path: "longcatvideocontinuation" in path.lower(),
            lambda path: "longcat" in path.lower(),
        ],
    )

    # StepVideo
    register_configs(
        sampling_param_cls=StepVideoT2VSamplingParam,
        pipeline_config_cls=StepVideoT2VConfig,
        hf_model_paths=[
            "FastVideo/stepvideo-t2v-diffusers",
        ],
        model_detectors=[lambda path: "stepvideo" in path.lower()],
    )

    # MatrixGame
    register_configs(
        sampling_param_cls=MatrixGame2_SamplingParam,
        pipeline_config_cls=MatrixGameI2V480PConfig,
        hf_model_paths=[
            "FastVideo/Matrix-Game-2.0-Base-Diffusers",
            "FastVideo/Matrix-Game-2.0-GTA-Diffusers",
            "FastVideo/Matrix-Game-2.0-TempleRun-Diffusers",
        ],
        model_detectors=[
            lambda path: "matrix-game" in path.lower() or "matrixgame" in path.
            lower(),
        ],
    )

    # Cosmos 2.5
    register_configs(
        sampling_param_cls=Cosmos25SamplingParamBase,
        pipeline_config_cls=Cosmos25Config,
        hf_model_paths=[
            "KyleShao/Cosmos-Predict2.5-2B-Diffusers",
        ],
        model_detectors=[
            lambda path: any(token in path.lower() for token in (
                "cosmos25",
                "cosmos2_5",
                "cosmos2.5",
            )),
        ],
    )

    # Cosmos 2
    register_configs(
        sampling_param_cls=Cosmos_Predict2_2B_Video2World_SamplingParam,
        pipeline_config_cls=CosmosConfig,
        hf_model_paths=[
            "nvidia/Cosmos-Predict2-2B-Video2World",
        ],
        model_detectors=[
            lambda path: "cosmos" in path.lower() and ("2.5" not in path.lower(
            ) and "2_5" not in path.lower() and "25" not in path.lower()),
        ],
    )

    # TurboDiffusion
    register_configs(
        sampling_param_cls=TurboDiffusionT2V_1_3B_SamplingParam,
        pipeline_config_cls=TurboDiffusionT2V_1_3B_Config,
        hf_model_paths=[
            "loayrashid/TurboWan2.1-T2V-1.3B-Diffusers",
        ],
        model_detectors=[
            lambda path: "turbodiffusion" in path.lower() or "turbowan" in path.
            lower()
        ],
    )
    register_configs(
        sampling_param_cls=TurboDiffusionT2V_14B_SamplingParam,
        pipeline_config_cls=TurboDiffusionT2V_14B_Config,
        hf_model_paths=[
            "loayrashid/TurboWan2.1-T2V-14B-Diffusers",
        ],
    )
    register_configs(
        sampling_param_cls=TurboDiffusionI2V_A14B_SamplingParam,
        pipeline_config_cls=TurboDiffusionI2V_A14B_Config,
        hf_model_paths=[
            "loayrashid/TurboWan2.2-I2V-A14B-Diffusers",
        ],
    )

    # Wan
    register_configs(
        sampling_param_cls=WanT2V_1_3B_SamplingParam,
        pipeline_config_cls=WanT2V480PConfig,
        hf_model_paths=[
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        ],
        model_detectors=[lambda path: "wanpipeline" in path.lower()],
    )
    register_configs(
        sampling_param_cls=WanT2V_14B_SamplingParam,
        pipeline_config_cls=WanT2V720PConfig,
        hf_model_paths=[
            "Wan-AI/Wan2.1-T2V-14B-Diffusers",
            "FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers",
        ],
    )
    register_configs(
        sampling_param_cls=WanI2V_14B_480P_SamplingParam,
        pipeline_config_cls=WanI2V480PConfig,
        hf_model_paths=[
            "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        ],
        model_detectors=[lambda path: "wanimagetovideo" in path.lower()],
    )
    register_configs(
        sampling_param_cls=WanI2V_14B_720P_SamplingParam,
        pipeline_config_cls=WanI2V720PConfig,
        hf_model_paths=[
            "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        ],
    )
    register_configs(
        sampling_param_cls=Wan2_1_Fun_1_3B_InP_SamplingParam,
        pipeline_config_cls=WanI2V480PConfig,
        hf_model_paths=[
            "weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers",
        ],
    )
    register_configs(
        sampling_param_cls=Wan2_1_Fun_1_3B_Control_SamplingParam,
        pipeline_config_cls=WANV2VConfig,
        hf_model_paths=[
            "IRMChen/Wan2.1-Fun-1.3B-Control-Diffusers",
        ],
    )
    register_configs(
        sampling_param_cls=FastWanT2V480P_SamplingParam,
        pipeline_config_cls=FastWan2_1_T2V_480P_Config,
        hf_model_paths=[
            "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
            "FastVideo/FastWan2.1-T2V-14B-480P-Diffusers",
        ],
        model_detectors=[lambda path: "wandmdpipeline" in path.lower()],
    )
    register_configs(
        sampling_param_cls=Wan2_2_TI2V_5B_SamplingParam,
        pipeline_config_cls=Wan2_2_TI2V_5B_Config,
        hf_model_paths=[
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        ],
    )
    register_configs(
        sampling_param_cls=Wan2_2_TI2V_5B_SamplingParam,
        pipeline_config_cls=FastWan2_2_TI2V_5B_Config,
        hf_model_paths=[
            "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
            "FastVideo/FastWan2.2-TI2V-5B-Diffusers",
        ],
    )
    register_configs(
        sampling_param_cls=Wan2_2_T2V_A14B_SamplingParam,
        pipeline_config_cls=Wan2_2_T2V_A14B_Config,
        hf_model_paths=[
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        ],
    )
    register_configs(
        sampling_param_cls=Wan2_2_I2V_A14B_SamplingParam,
        pipeline_config_cls=Wan2_2_I2V_A14B_Config,
        hf_model_paths=[
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        ],
    )
    register_configs(
        sampling_param_cls=SelfForcingWan2_1_T2V_1_3B_480P_SamplingParam,
        pipeline_config_cls=SelfForcingWanT2V480PConfig,
        hf_model_paths=[
            "wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers",
        ],
        model_detectors=[lambda path: "wancausaldmdpipeline" in path.lower()],
    )
    register_configs(
        sampling_param_cls=SelfForcingWan2_2_T2V_A14B_480P_SamplingParam,
        pipeline_config_cls=SelfForcingWan2_2_T2V480PConfig,
        hf_model_paths=[
            "rand0nmr/SFWan2.2-T2V-A14B-Diffusers",
            "FastVideo/SFWan2.2-I2V-A14B-Preview-Diffusers",
        ],
    )

    # SD3.5
    register_configs(
        sampling_param_cls=SD35SamplingParam,
        pipeline_config_cls=SD35Config,
        hf_model_paths=[
            "stabilityai/stable-diffusion-3.5-medium",
        ],
        model_detectors=[
            lambda path: any(token in path.lower() for token in (
                "sd35",
                "stablediffusion3",
                "stabilityai__stable-diffusion-3.5-medium",
            )),
        ],
    )

    # Lucy-Edit (decart-ai video editing)
    register_configs(
        sampling_param_cls=LucyEditSamplingParam,
        pipeline_config_cls=LucyEditConfig,
        hf_model_paths=[
            "decart-ai/Lucy-Edit-Dev",
        ],
        model_detectors=[
            lambda path: "lucyeditpipeline" in path.lower() or "lucy-edit" in
            path.lower(),
        ],
    )


# --- Part 3: Main Resolver ---


@dataclasses.dataclass
class ModelInfo:
    pipeline_cls: type[ComposedPipelineBase]
    sampling_param_cls: type[SamplingParam]
    pipeline_config_cls: type[PipelineConfig]


@lru_cache(maxsize=32)
def get_model_info(
    model_path: str,
    pipeline_type: PipelineType | str | None = None,
    workload_type: WorkloadType | None = None,
    override_pipeline_cls_name: str | None = None,
) -> ModelInfo:
    from fastvideo.pipelines.pipeline_registry import (PipelineType,
                                                       get_pipeline_registry)

    if pipeline_type is None:
        pipeline_type = PipelineType.BASIC
    elif isinstance(pipeline_type, str):
        pipeline_type = PipelineType.from_string(pipeline_type)

    if workload_type is None:
        workload_type = WorkloadType.T2V

    if os.path.exists(model_path):
        config = verify_model_config_and_directory(model_path)
    else:
        config = maybe_download_model_index(model_path)

    pipeline_name = config.get("_class_name")
    if override_pipeline_cls_name:
        logger.info("Overriding pipeline class name from %s to %s",
                    pipeline_name, override_pipeline_cls_name)
        pipeline_name = override_pipeline_cls_name

    if pipeline_name is None:
        raise ValueError(
            "Model config does not contain a _class_name attribute. "
            "Only diffusers format is supported.")

    pipeline_registry = get_pipeline_registry(pipeline_type)
    pipeline_cls = pipeline_registry.resolve_pipeline_cls(
        pipeline_name, pipeline_type, workload_type)

    config_info = _get_config_info(model_path, raise_on_missing=True)
    assert config_info is not None, "config_info must be resolved"

    sampling_param_cls = config_info.sampling_param_cls or SamplingParam

    return ModelInfo(
        pipeline_cls=pipeline_cls,
        sampling_param_cls=sampling_param_cls,
        pipeline_config_cls=config_info.pipeline_config_cls,
    )


def get_pipeline_config_cls_from_name(
        pipeline_name_or_path: str) -> type[PipelineConfig]:
    config_info = _get_config_info(pipeline_name_or_path,
                                   raise_on_missing=False)
    if config_info is None:
        raise ValueError(
            f"No match found for pipeline {pipeline_name_or_path}, please check the pipeline name or path."
        )
    return config_info.pipeline_config_cls


def get_sampling_param_cls_for_name(pipeline_name_or_path: str) -> Any | None:
    config_info = _get_config_info(pipeline_name_or_path,
                                   raise_on_missing=False)
    if config_info is None:
        logger.warning(
            "No match found for pipeline %s, using default sampling param.",
            pipeline_name_or_path)
        return None
    return config_info.sampling_param_cls


_register_configs()

__all__ = [
    "ConfigInfo",
    "ModelInfo",
    "get_model_info",
    "get_pipeline_config_cls_from_name",
    "get_sampling_param_cls_for_name",
    "get_pipeline_config_classes",
]
