# SPDX-License-Identifier: Apache-2.0
import os
from collections.abc import Callable
from typing import Any

from fastvideo.configs.sample.hunyuan import (FastHunyuanSamplingParam,
                                              HunyuanSamplingParam)
from fastvideo.configs.sample.hunyuan15 import Hunyuan15_480P_SamplingParam, Hunyuan15_720P_SamplingParam
from fastvideo.configs.sample.stepvideo import StepVideoT2VSamplingParam

from fastvideo.configs.sample.cosmos import Cosmos_Predict2_2B_Video2World_SamplingParam
from fastvideo.configs.sample.cosmos2_5 import Cosmos_Predict2_5_2B_Diffusers_SamplingParam
from fastvideo.configs.sample.ltx2 import LTX2SamplingParam

# isort: off
from fastvideo.configs.sample.wan import (
    FastWanT2V480P_SamplingParam,
    Wan2_1_Fun_1_3B_InP_SamplingParam,
    Wan2_2_I2V_A14B_SamplingParam,
    Wan2_2_T2V_A14B_SamplingParam,
    Wan2_2_TI2V_5B_SamplingParam,
    WanI2V_14B_480P_SamplingParam,
    WanI2V_14B_720P_SamplingParam,
    WanT2V_1_3B_SamplingParam,
    WanT2V_14B_SamplingParam,
    Wan2_1_Fun_1_3B_Control_SamplingParam,
    SelfForcingWan2_1_T2V_1_3B_480P_SamplingParam,
    SelfForcingWan2_2_T2V_A14B_480P_SamplingParam,
    MatrixGame2_SamplingParam,
)
from fastvideo.configs.sample.turbodiffusion import (
    TurboDiffusionT2V_1_3B_SamplingParam,
    TurboDiffusionT2V_14B_SamplingParam,
    TurboDiffusionI2V_A14B_SamplingParam,
)
# isort: on
from fastvideo.logger import init_logger
from fastvideo.utils import (maybe_download_model_index,
                             verify_model_config_and_directory)

logger = init_logger(__name__)
# Registry maps specific model weights to their config classes
SAMPLING_PARAM_REGISTRY: dict[str, Any] = {
    "FastVideo/FastHunyuan-diffusers": FastHunyuanSamplingParam,
    "hunyuanvideo-community/HunyuanVideo": HunyuanSamplingParam,
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v":
    Hunyuan15_480P_SamplingParam,
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v":
    Hunyuan15_720P_SamplingParam,
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
    "KyleShao/Cosmos-Predict2.5-2B-Diffusers":
    Cosmos_Predict2_5_2B_Diffusers_SamplingParam,

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

    # Add other specific weight variants
}

# For determining pipeline type from model ID
SAMPLING_PARAM_DETECTOR: dict[str, Callable[[str], bool]] = {
    "hunyuan":
    lambda id: "hunyuan" in id.lower(),
    "hunyuan15":
    lambda id: "hunyuan15" in id.lower(),
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
    # Add other pipeline architecture detectors
}

# Fallback configs when exact match isn't found but architecture is detected
SAMPLING_FALLBACK_PARAM: dict[str, Any] = {
    "hunyuan":
    HunyuanSamplingParam,  # Base Hunyuan config as fallback for any Hunyuan variant
    "hunyuan15":
    Hunyuan15_480P_SamplingParam,  # Base Hunyuan15 config as fallback for any Hunyuan15 variant
    "wanpipeline":
    WanT2V_1_3B_SamplingParam,  # Base Wan config as fallback for any Wan variant
    "wanimagetovideo": WanI2V_14B_480P_SamplingParam,
    "wandmdpipeline": FastWanT2V480P_SamplingParam,
    "wancausaldmdpipeline": SelfForcingWan2_1_T2V_1_3B_480P_SamplingParam,
    "stepvideo": StepVideoT2VSamplingParam,
    "matrixgame": MatrixGame2_SamplingParam,
    "turbodiffusion":
    TurboDiffusionT2V_1_3B_SamplingParam,  # Default to T2V for fallback
    "cosmos25": Cosmos_Predict2_5_2B_Diffusers_SamplingParam,
    "cosmos": Cosmos_Predict2_2B_Video2World_SamplingParam,
    "ltx2": LTX2SamplingParam,
    # Other fallbacks by architecture
}


def get_sampling_param_cls_for_name(pipeline_name_or_path: str) -> Any | None:
    """Get the appropriate sampling param for specific pretrained weights."""

    # First try exact match for specific weights
    if pipeline_name_or_path in SAMPLING_PARAM_REGISTRY:
        return SAMPLING_PARAM_REGISTRY[pipeline_name_or_path]

    # Try partial matches (for local paths that might include the weight ID)
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

    # If no match, try to use the fallback config
    fallback_config = None
    # Try to determine pipeline architecture for fallback
    for pipeline_type, detector in SAMPLING_PARAM_DETECTOR.items():
        if detector(pipeline_name.lower()):
            fallback_config = SAMPLING_FALLBACK_PARAM.get(pipeline_type)
            break

    logger.warning(
        "No match found for pipeline %s, using fallback sampling param %s.",
        pipeline_name_or_path, fallback_config)
    return fallback_config
