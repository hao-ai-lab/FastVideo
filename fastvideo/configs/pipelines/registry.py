# SPDX-License-Identifier: Apache-2.0
"""Registry for pipeline weight-specific configurations."""

import os
from collections.abc import Callable

from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.configs.pipelines.cosmos import CosmosConfig
from fastvideo.configs.pipelines.cosmos2_5 import Cosmos25Config
from fastvideo.configs.pipelines.hunyuan import FastHunyuanConfig, HunyuanConfig
from fastvideo.configs.pipelines.hunyuan15 import Hunyuan15T2V480PConfig, Hunyuan15T2V720PConfig
from fastvideo.configs.pipelines.ltx2 import LTX2T2VConfig
from fastvideo.configs.pipelines.stepvideo import StepVideoT2VConfig
from fastvideo.configs.pipelines.longcat import LongCatT2V480PConfig
from fastvideo.configs.pipelines.turbodiffusion import (
    TurboDiffusionT2V_1_3B_Config, TurboDiffusionT2V_14B_Config,
    TurboDiffusionI2V_A14B_Config)

# isort: off
from fastvideo.configs.pipelines.wan import (
    FastWan2_1_T2V_480P_Config, FastWan2_2_TI2V_5B_Config,
    Wan2_2_I2V_A14B_Config, Wan2_2_T2V_A14B_Config, Wan2_2_TI2V_5B_Config,
    WanI2V480PConfig, WanI2V720PConfig, WanT2V480PConfig, WanT2V720PConfig,
    SelfForcingWanT2V480PConfig, WANV2VConfig, SelfForcingWan2_2_T2V480PConfig,
    MatrixGameI2V480PConfig)
# isort: on
from fastvideo.logger import init_logger
from fastvideo.utils import (maybe_download_model_index,
                             verify_model_config_and_directory)

logger = init_logger(__name__)

# Registry maps specific model weights to their config classes
PIPE_NAME_TO_CONFIG: dict[str, type[PipelineConfig]] = {
    "FastVideo/FastHunyuan-diffusers": FastHunyuanConfig,
    "hunyuanvideo-community/HunyuanVideo": HunyuanConfig,
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v":
    Hunyuan15T2V480PConfig,
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v":
    Hunyuan15T2V720PConfig,
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
    # Add other specific weight variants
}

# For determining pipeline type from model ID
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
    # Add other pipeline architecture detectors
}

# Fallback configs when exact match isn't found but architecture is detected
PIPELINE_FALLBACK_CONFIG: dict[str, type[PipelineConfig]] = {
    "longcatimagetovideo": LongCatT2V480PConfig,
    "longcatvideocontinuation": LongCatT2V480PConfig,
    "longcat": LongCatT2V480PConfig,
    "cosmos25": Cosmos25Config,
    "hunyuan":
    HunyuanConfig,  # Base Hunyuan config as fallback for any Hunyuan variant
    "matrixgame": MatrixGameI2V480PConfig,
    "hunyuan15":
    Hunyuan15T2V480PConfig,  # Base Hunyuan15 config as fallback for any Hunyuan15 variant
    "wanpipeline":
    WanT2V480PConfig,  # Base Wan config as fallback for any Wan variant
    "wanimagetovideo": WanI2V480PConfig,
    "wandmdpipeline": FastWan2_1_T2V_480P_Config,
    "wancausaldmdpipeline": SelfForcingWanT2V480PConfig,
    "stepvideo": StepVideoT2VConfig,
    "turbodiffusion": TurboDiffusionT2V_1_3B_Config,
    "ltx2": LTX2T2VConfig,
    # Other fallbacks by architecture
}


def get_pipeline_config_cls_from_name(
        pipeline_name_or_path: str) -> type[PipelineConfig]:
    """Get the appropriate configuration class for a given pipeline name or path.

    This function implements a multi-step lookup process to find the most suitable
    configuration class for a given pipeline. It follows this order:
    1. Exact match in the PIPE_NAME_TO_CONFIG
    2. Partial match in the PIPE_NAME_TO_CONFIG
    3. Fallback to class name in the model_index.json
    4. else raise an error

    Args:
        pipeline_name_or_path (str): The name or path of the pipeline. This can be:
            - A registered model ID (e.g., "FastVideo/FastHunyuan-diffusers")
            - A local path to a model directory
            - A model ID that will be downloaded

    Returns:
        Type[PipelineConfig]: The configuration class that best matches the pipeline.
            This will be one of:
            - A specific weight configuration class if an exact match is found
            - A fallback configuration class based on the pipeline architecture
            - The base PipelineConfig class if no matches are found

    Note:
        - For local paths, the function will verify the model configuration
        - For remote models, it will attempt to download the model index
        - Warning messages are logged when falling back to less specific configurations
    """

    pipeline_config_cls: type[PipelineConfig] | None = None

    # First try exact match for specific weights
    if pipeline_name_or_path in PIPE_NAME_TO_CONFIG:
        pipeline_config_cls = PIPE_NAME_TO_CONFIG[pipeline_name_or_path]
        return pipeline_config_cls

    # Try partial matches (for local paths that might include the weight ID)
    for registered_id, config_class in PIPE_NAME_TO_CONFIG.items():
        if registered_id in pipeline_name_or_path:
            pipeline_config_cls = config_class
            break

    # If no match, try to use the fallback config
    if pipeline_config_cls is None:
        if os.path.exists(pipeline_name_or_path):
            config = verify_model_config_and_directory(pipeline_name_or_path)
        else:
            config = maybe_download_model_index(pipeline_name_or_path)
        logger.warning(
            "Trying to use the config from the model_index.json. FastVideo may not correctly identify the optimal config for this model in this situation."
        )

        pipeline_name = config["_class_name"]
        # Try to determine pipeline architecture for fallback
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
