# SPDX-License-Identifier: Apache-2.0
import os
from typing import Any, Callable, Dict, Optional

from fastvideo.v1.configs.sample.hunyuan import (FastHunyuanSamplingParam,
                                                 HunyuanSamplingParam)
from fastvideo.v1.configs.sample.stepvideo import StepVideoT2VSamplingParam
from fastvideo.v1.configs.sample.wan import (WanI2V_14B_480P_SamplingParam,
                                             WanI2V_14B_720P_SamplingParam,
                                             WanT2V_1_3B_SamplingParam,
                                             WanT2V_14B_SamplingParam)
from fastvideo.v1.logger import init_logger
from fastvideo.v1.utils import (maybe_download_model_index,
                                verify_model_config_and_directory)

logger = init_logger(__name__)
# Registry maps specific model weights to their config classes
SAMPLING_PARAM_REGISTRY: Dict[str, Any] = {
    "FastVideo/FastHunyuan-diffusers": FastHunyuanSamplingParam,
    "hunyuanvideo-community/HunyuanVideo": HunyuanSamplingParam,
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": WanT2V_1_3B_SamplingParam,
    "Wan-AI/Wan2.1-T2V-14B-Diffusers": WanT2V_14B_SamplingParam,
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers": WanI2V_14B_480P_SamplingParam,
    "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers": WanI2V_14B_720P_SamplingParam,
    "FastVideo/stepvideo-t2v-diffusers": StepVideoT2VSamplingParam,
    # Add other specific weight variants
}

# For determining pipeline type from model ID
SAMPLING_PARAM_DETECTOR: Dict[str, Callable[[str], bool]] = {
    "hunyuan": lambda id: "hunyuan" in id.lower(),
    "wanpipeline": lambda id: "wanpipeline" in id.lower(),
    "wanimagetovideo": lambda id: "wanimagetovideo" in id.lower(),
    "stepvideo": lambda id: "stepvideo" in id.lower(),
    # Add other pipeline architecture detectors
}

# Fallback configs when exact match isn't found but architecture is detected
SAMPLING_FALLBACK_PARAM: Dict[str, Any] = {
    "hunyuan":
    HunyuanSamplingParam,  # Base Hunyuan config as fallback for any Hunyuan variant
    "wanpipeline":
    WanT2V_1_3B_SamplingParam,  # Base Wan config as fallback for any Wan variant
    "wanimagetovideo": WanI2V_14B_480P_SamplingParam,
    "stepvideo": StepVideoT2VSamplingParam
    # Other fallbacks by architecture
}


def get_sampling_param_cls_for_name(
        pipeline_name_or_path: str) -> Optional[Any]:
    """Get the appropriate sampling param for specific pretrained weights."""

    if os.path.exists(pipeline_name_or_path):
        config = verify_model_config_and_directory(pipeline_name_or_path)
        logger.warning(
            "FastVideo may not correctly identify the optimal sampling param for this model, as the local directory may have been renamed."
        )
    else:
        config = maybe_download_model_index(pipeline_name_or_path)

    pipeline_name = config["_class_name"]

    # First try exact match for specific weights
    if pipeline_name_or_path in SAMPLING_PARAM_REGISTRY:
        return SAMPLING_PARAM_REGISTRY[pipeline_name_or_path]

    # Try partial matches (for local paths that might include the weight ID)
    for registered_id, config_class in SAMPLING_PARAM_REGISTRY.items():
        if registered_id in pipeline_name_or_path:
            return config_class

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
