# SPDX-License-Identifier: Apache-2.0
"""
Central registry for FastVideo pipelines and model configuration discovery.

This module unifies pipeline config and sampling param resolution, and mirrors
sglang's improved registry approach. It keeps the legacy behavior but provides
one entrypoint to resolve all model metadata.
"""

from __future__ import annotations

import dataclasses
import os
from functools import lru_cache
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.configs.pipelines.cosmos import CosmosConfig
from fastvideo.configs.pipelines.cosmos2_5 import Cosmos25Config
from fastvideo.configs.pipelines.hunyuan import FastHunyuanConfig, HunyuanConfig
from fastvideo.configs.pipelines.hunyuan15 import (Hunyuan15T2V480PConfig,
                                                   Hunyuan15T2V720PConfig)
from fastvideo.configs.pipelines.hyworld import HYWorldConfig
from fastvideo.configs.pipelines.longcat import LongCatT2V480PConfig
from fastvideo.configs.pipelines.ltx2 import LTX2T2VConfig
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
from fastvideo.configs.sample.base import SamplingParam
from fastvideo.configs.sample.cosmos import (
    Cosmos_Predict2_2B_Video2World_SamplingParam, )
from fastvideo.configs.sample.cosmos2_5 import Cosmos25SamplingParamBase
from fastvideo.configs.sample.hunyuan import (FastHunyuanSamplingParam,
                                              HunyuanSamplingParam)
from fastvideo.configs.sample.hunyuan15 import (Hunyuan15_480P_SamplingParam,
                                                Hunyuan15_720P_SamplingParam)
from fastvideo.configs.sample.hyworld import HYWorld_SamplingParam
from fastvideo.configs.sample.ltx2 import LTX2SamplingParam
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

PIPE_NAME_TO_CONFIG: dict[str, type[PipelineConfig]] = {
    "FastVideo/FastHunyuan-diffusers": FastHunyuanConfig,
    "hunyuanvideo-community/HunyuanVideo": HunyuanConfig,
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v":
    Hunyuan15T2V480PConfig,
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v":
    Hunyuan15T2V720PConfig,
    "FastVideo/HY-WorldPlay-Bidirectional-Diffusers": HYWorldConfig,
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": WanT2V480PConfig,
    "weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers": WanI2V480PConfig,
    "IRMChen/Wan2.1-Fun-1.3B-Control-Diffusers": WANV2VConfig,
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers": WanI2V480PConfig,
    "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers": WanI2V720PConfig,
    "Wan-AI/Wan2.1-T2V-14B-Diffusers": WanT2V720PConfig,
    "FastVideo/FastWan2.1-T2V-1.3B-Diffusers": FastWan2_1_T2V_480P_Config,
    "FastVideo/FastWan2.1-T2V-14B-480P-Diffusers": FastWan2_1_T2V_480P_Config,
    "FastVideo/FastWan2.2-TI2V-5B-Diffusers": FastWan2_2_TI2V_5B_Config,
    "FastVideo/stepvideo-t2v-diffusers": StepVideoT2VConfig,
    "FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers": WanT2V720PConfig,
    "wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers": SelfForcingWanT2V480PConfig,
    "rand0nmr/SFWan2.2-T2V-A14B-Diffusers": SelfForcingWan2_2_T2V480PConfig,
    "FastVideo/SFWan2.2-I2V-A14B-Preview-Diffusers":
    SelfForcingWan2_2_T2V480PConfig,
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers": Wan2_2_TI2V_5B_Config,
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": Wan2_2_T2V_A14B_Config,
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers": Wan2_2_I2V_A14B_Config,
    "nvidia/Cosmos-Predict2-2B-Video2World": CosmosConfig,
    "KyleShao/Cosmos-Predict2.5-2B-Diffusers": Cosmos25Config,
    "FastVideo/Matrix-Game-2.0-Base-Diffusers": MatrixGameI2V480PConfig,
    "FastVideo/Matrix-Game-2.0-GTA-Diffusers": MatrixGameI2V480PConfig,
    "FastVideo/Matrix-Game-2.0-TempleRun-Diffusers": MatrixGameI2V480PConfig,
    # LongCat Video models
    "FastVideo/LongCat-Video-T2V-Diffusers": LongCatT2V480PConfig,
    "FastVideo/LongCat-Video-I2V-Diffusers": LongCatT2V480PConfig,
    "FastVideo/LongCat-Video-VC-Diffusers": LongCatT2V480PConfig,
    # LTX-2 models
    "Lightricks/LTX-2": LTX2T2VConfig,
    "converted/ltx2_diffusers": LTX2T2VConfig,
    # TurboDiffusion models
    "loayrashid/TurboWan2.1-T2V-1.3B-Diffusers": TurboDiffusionT2V_1_3B_Config,
    "loayrashid/TurboWan2.1-T2V-14B-Diffusers": TurboDiffusionT2V_14B_Config,
    "loayrashid/TurboWan2.2-I2V-A14B-Diffusers": TurboDiffusionI2V_A14B_Config,
}

PIPELINE_DETECTOR: dict[str, Callable[[str], bool]] = {
    "longcatimagetovideo":
    lambda id: "longcatimagetovideo" in id.lower(),
    "longcatvideocontinuation":
    lambda id: "longcatvideocontinuation" in id.lower(),
    "longcat":
    lambda id: "longcat" in id.lower(),
    "hunyuan":
    lambda id: "hunyuan" in id.lower(),
    "hunyuan15":
    lambda id: "hunyuan15" in id.lower(),
    "hyworld":
    lambda id: "hyworld" in id.lower(),
    "matrixgame":
    lambda id: "matrix-game" in id.lower() or "matrixgame" in id.lower(),
    "wanpipeline":
    lambda id: "wanpipeline" in id.lower(),
    "wanimagetovideo":
    lambda id: "wanimagetovideo" in id.lower(),
    "wandmdpipeline":
    lambda id: "wandmdpipeline" in id.lower(),
    "wancausaldmdpipeline":
    lambda id: "wancausaldmdpipeline" in id.lower(),
    "stepvideo":
    lambda id: "stepvideo" in id.lower(),
    "cosmos":
    lambda id: "cosmos" in id.lower() and ("2.5" not in id.lower(
    ) and "2_5" not in id.lower() and "25" not in id.lower()),
    "cosmos25":
    lambda id: "cosmos25" in id.lower(),
    "turbodiffusion":
    lambda id: "turbodiffusion" in id.lower() or "turbowan" in id.lower(),
    "ltx2":
    lambda id: "ltx2" in id.lower() or "ltx-2" in id.lower(),
}

PIPELINE_FALLBACK_CONFIG: dict[str, type[PipelineConfig]] = {
    "longcatimagetovideo": LongCatT2V480PConfig,
    "longcatvideocontinuation": LongCatT2V480PConfig,
    "longcat": LongCatT2V480PConfig,
    "cosmos25": Cosmos25Config,
    "hunyuan": HunyuanConfig,
    "matrixgame": MatrixGameI2V480PConfig,
    "hunyuan15": Hunyuan15T2V480PConfig,
    "hyworld": HYWorldConfig,
    "wanpipeline": WanT2V480PConfig,
    "wanimagetovideo": WanI2V480PConfig,
    "wandmdpipeline": FastWan2_1_T2V_480P_Config,
    "wancausaldmdpipeline": SelfForcingWanT2V480PConfig,
    "stepvideo": StepVideoT2VConfig,
    "turbodiffusion": TurboDiffusionT2V_1_3B_Config,
    "ltx2": LTX2T2VConfig,
}

SAMPLING_PARAM_REGISTRY: dict[str, Any] = {
    "FastVideo/FastHunyuan-diffusers": FastHunyuanSamplingParam,
    "hunyuanvideo-community/HunyuanVideo": HunyuanSamplingParam,
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v":
    Hunyuan15_480P_SamplingParam,
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v":
    Hunyuan15_720P_SamplingParam,
    "FastVideo/HY-WorldPlay-Bidirectional-Diffusers": HYWorld_SamplingParam,
    "FastVideo/stepvideo-t2v-diffusers": StepVideoT2VSamplingParam,
    # Wan2.1
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": WanT2V_1_3B_SamplingParam,
    "Wan-AI/Wan2.1-T2V-14B-Diffusers": WanT2V_14B_SamplingParam,
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers": WanI2V_14B_480P_SamplingParam,
    "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers": WanI2V_14B_720P_SamplingParam,
    "weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers":
    Wan2_1_Fun_1_3B_InP_SamplingParam,
    "IRMChen/Wan2.1-Fun-1.3B-Control-Diffusers":
    Wan2_1_Fun_1_3B_Control_SamplingParam,
    # Wan2.2
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers": Wan2_2_TI2V_5B_SamplingParam,
    "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers":
    Wan2_2_TI2V_5B_SamplingParam,
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": Wan2_2_T2V_A14B_SamplingParam,
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers": Wan2_2_I2V_A14B_SamplingParam,
    # FastWan2.1
    "FastVideo/FastWan2.1-T2V-1.3B-Diffusers": FastWanT2V480P_SamplingParam,
    # FastWan2.2
    "FastVideo/FastWan2.2-TI2V-5B-Diffusers": Wan2_2_TI2V_5B_SamplingParam,
    # Causal Self-Forcing Wan2.1
    "wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers":
    SelfForcingWan2_1_T2V_1_3B_480P_SamplingParam,
    # Causal Self-Forcing Wan2.2
    "rand0nmr/SFWan2.2-T2V-A14B-Diffusers":
    SelfForcingWan2_2_T2V_A14B_480P_SamplingParam,
    "FastVideo/SFWan2.2-I2V-A14B-Preview-Diffusers":
    SelfForcingWan2_2_T2V_A14B_480P_SamplingParam,
    # Cosmos2
    "nvidia/Cosmos-Predict2-2B-Video2World":
    Cosmos_Predict2_2B_Video2World_SamplingParam,
    # Cosmos2.5
    "KyleShao/Cosmos-Predict2.5-2B-Diffusers": Cosmos25SamplingParamBase,
    # MatrixGame2.0 models
    "FastVideo/Matrix-Game-2.0-Base-Diffusers": MatrixGame2_SamplingParam,
    "FastVideo/Matrix-Game-2.0-GTA-Diffusers": MatrixGame2_SamplingParam,
    "FastVideo/Matrix-Game-2.0-TempleRun-Diffusers": MatrixGame2_SamplingParam,
    # TurboDiffusion models
    "loayrashid/TurboWan2.1-T2V-1.3B-Diffusers":
    TurboDiffusionT2V_1_3B_SamplingParam,
    "loayrashid/TurboWan2.1-T2V-14B-Diffusers":
    TurboDiffusionT2V_14B_SamplingParam,
    "loayrashid/TurboWan2.2-I2V-A14B-Diffusers":
    TurboDiffusionI2V_A14B_SamplingParam,
    # LTX-2 models
    "Lightricks/LTX-2": LTX2SamplingParam,
    "FastVideo/LTX2-Distilled-Diffusers": LTX2SamplingParam,
}

SAMPLING_PARAM_DETECTOR: dict[str, Callable[[str], bool]] = {
    "hunyuan":
    lambda id: "hunyuan" in id.lower(),
    "hunyuan15":
    lambda id: "hunyuan15" in id.lower(),
    "hyworld":
    lambda id: "hyworld" in id.lower(),
    "wanpipeline":
    lambda id: "wanpipeline" in id.lower(),
    "wanimagetovideo":
    lambda id: "wanimagetovideo" in id.lower(),
    "stepvideo":
    lambda id: "stepvideo" in id.lower(),
    "wandmdpipeline":
    lambda id: "wandmdpipeline" in id.lower(),
    "wancausaldmdpipeline":
    lambda id: "wancausaldmdpipeline" in id.lower(),
    "matrixgame":
    lambda id: "matrixgame" in id.lower() or "matrix-game" in id.lower(),
    "turbodiffusion":
    lambda id: "turbodiffusion" in id.lower() or "turbowan" in id.lower(),
    "cosmos25":
    lambda id: "cosmos2_5" in id.lower(),
    "cosmos":
    lambda id: "cosmos" in id.lower() and "2_5" not in id.lower(),
    "ltx2":
    lambda id: "ltx2" in id.lower() or "ltx-2" in id.lower(),
}

SAMPLING_FALLBACK_PARAM: dict[str, Any] = {
    "hunyuan": HunyuanSamplingParam,
    "hunyuan15": Hunyuan15_480P_SamplingParam,
    "hyworld": HYWorld_SamplingParam,
    "wanpipeline": WanT2V_1_3B_SamplingParam,
    "wanimagetovideo": WanI2V_14B_480P_SamplingParam,
    "wandmdpipeline": FastWanT2V480P_SamplingParam,
    "wancausaldmdpipeline": SelfForcingWan2_1_T2V_1_3B_480P_SamplingParam,
    "stepvideo": StepVideoT2VSamplingParam,
    "matrixgame": MatrixGame2_SamplingParam,
    "turbodiffusion": TurboDiffusionT2V_1_3B_SamplingParam,
    "cosmos25": Cosmos25SamplingParamBase,
    "cosmos": Cosmos_Predict2_2B_Video2World_SamplingParam,
    "ltx2": LTX2SamplingParam,
}


def get_pipeline_config_cls_from_name(
        pipeline_name_or_path: str) -> type[PipelineConfig]:
    pipeline_config_cls: type[PipelineConfig] | None = None

    if pipeline_name_or_path in PIPE_NAME_TO_CONFIG:
        return PIPE_NAME_TO_CONFIG[pipeline_name_or_path]

    for registered_id, config_class in PIPE_NAME_TO_CONFIG.items():
        if registered_id in pipeline_name_or_path:
            pipeline_config_cls = config_class
            break

    if pipeline_config_cls is None:
        if os.path.exists(pipeline_name_or_path):
            config = verify_model_config_and_directory(pipeline_name_or_path)
        else:
            config = maybe_download_model_index(pipeline_name_or_path)
        logger.warning(
            "Trying to use the config from the model_index.json. FastVideo may not correctly identify the optimal config for this model in this situation."
        )

        pipeline_name = config["_class_name"]
        for pipeline_type, detector in PIPELINE_DETECTOR.items():
            if detector(pipeline_name.lower()):
                pipeline_config_cls = PIPELINE_FALLBACK_CONFIG.get(
                    pipeline_type)
                break

        if pipeline_config_cls is not None:
            logger.warning(
                "No match found for pipeline %s, using fallback config %s.",
                pipeline_name_or_path, pipeline_config_cls)

    if pipeline_config_cls is None:
        raise ValueError(
            f"No match found for pipeline {pipeline_name_or_path}, please check the pipeline name or path."
        )

    return pipeline_config_cls


def get_sampling_param_cls_for_name(pipeline_name_or_path: str) -> Any | None:
    if pipeline_name_or_path in SAMPLING_PARAM_REGISTRY:
        return SAMPLING_PARAM_REGISTRY[pipeline_name_or_path]

    for registered_id, config_class in SAMPLING_PARAM_REGISTRY.items():
        if registered_id in pipeline_name_or_path:
            return config_class

    matrixgame_patterns = ["Matrix-Game", "Skywork--Matrix-Game", "matrixgame"]
    for pattern in matrixgame_patterns:
        if pattern.lower() in pipeline_name_or_path.lower():
            return MatrixGame2_SamplingParam

    if os.path.exists(pipeline_name_or_path):
        config = verify_model_config_and_directory(pipeline_name_or_path)
    else:
        config = maybe_download_model_index(pipeline_name_or_path)

    pipeline_name = config["_class_name"]

    fallback_config = None
    for pipeline_type, detector in SAMPLING_PARAM_DETECTOR.items():
        if detector(pipeline_name.lower()):
            fallback_config = SAMPLING_FALLBACK_PARAM.get(pipeline_type)
            break

    logger.warning(
        "No match found for pipeline %s, using fallback sampling param %s.",
        pipeline_name_or_path, fallback_config)
    return fallback_config


@dataclasses.dataclass
class ConfigInfo:
    pipeline_config_cls: type[PipelineConfig]
    sampling_param_cls: type[SamplingParam] | None


def get_config_info(model_path: str) -> ConfigInfo:
    pipeline_config_cls = get_pipeline_config_cls_from_name(model_path)
    sampling_param_cls = get_sampling_param_cls_for_name(model_path)
    return ConfigInfo(
        pipeline_config_cls=pipeline_config_cls,
        sampling_param_cls=sampling_param_cls,
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

    config_info = get_config_info(model_path)
    sampling_param_cls = config_info.sampling_param_cls or SamplingParam

    return ModelInfo(
        pipeline_cls=pipeline_cls,
        sampling_param_cls=sampling_param_cls,
        pipeline_config_cls=config_info.pipeline_config_cls,
    )


__all__ = [
    "ConfigInfo",
    "ModelInfo",
    "get_model_info",
    "get_config_info",
    "get_pipeline_config_cls_from_name",
    "get_sampling_param_cls_for_name",
]
