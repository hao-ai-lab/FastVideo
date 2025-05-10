"""Registry for pipeline weight-specific configurations."""

import os
from typing import Callable, Dict, Optional, Type

from fastvideo.v1.configs.pipelines.base import PipelineConfig
from fastvideo.v1.configs.pipelines.hunyuan import (FastHunyuanConfig,
                                                    HunyuanConfig)
from fastvideo.v1.configs.pipelines.stepvideo import StepVideoT2VConfig
from fastvideo.v1.configs.pipelines.wan import (WanI2V480PConfig,
                                                WanI2V720PConfig,
                                                WanT2V480PConfig,
                                                WanT2V720PConfig)
from fastvideo.v1.logger import init_logger
from fastvideo.v1.utils import (maybe_download_model_index,
                                verify_model_config_and_directory)

logger = init_logger(__name__)

# Registry maps specific model weights to their config classes
WEIGHT_CONFIG_REGISTRY: Dict[str, Type[PipelineConfig]] = {
    "FastVideo/FastHunyuan-diffusers": FastHunyuanConfig,
    "hunyuanvideo-community/HunyuanVideo": HunyuanConfig,
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": WanT2V480PConfig,
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers": WanI2V480PConfig,
    "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers": WanI2V720PConfig,
    "Wan-AI/Wan2.1-T2V-14B-Diffusers": WanT2V720PConfig,
    "FastVideo/stepvideo-t2v-diffusers": StepVideoT2VConfig,
    # Add other specific weight variants
}

# For determining pipeline type from model ID
PIPELINE_DETECTOR: Dict[str, Callable[[str], bool]] = {
    "hunyuan": lambda id: "hunyuan" in id.lower(),
    "wanpipeline": lambda id: "wanpipeline" in id.lower(),
    "wanimagetovideo": lambda id: "wanimagetovideo" in id.lower(),
    "stepvideo": lambda id: "stepvideo" in id.lower(),
    # Add other pipeline architecture detectors
}

# Fallback configs when exact match isn't found but architecture is detected
PIPELINE_FALLBACK_CONFIG: Dict[str, Type[PipelineConfig]] = {
    "hunyuan":
    HunyuanConfig,  # Base Hunyuan config as fallback for any Hunyuan variant
    "wanpipeline":
    WanT2V480PConfig,  # Base Wan config as fallback for any Wan variant
    "wanimagetovideo": WanI2V480PConfig,
    "stepvideo": StepVideoT2VConfig
    # Other fallbacks by architecture
}


def get_pipeline_config_cls_for_name(
        pipeline_name_or_path: str) -> Optional[type[PipelineConfig]]:
    """Get the appropriate config class for specific pretrained weights."""

    if os.path.exists(pipeline_name_or_path):
        config = verify_model_config_and_directory(pipeline_name_or_path)
        logger.warning(
            "FastVideo may not correctly identify the optimal config for this model, as the local directory may have been renamed."
        )
    else:
        config = maybe_download_model_index(pipeline_name_or_path)

    pipeline_name = config["_class_name"]

    # First try exact match for specific weights
    if pipeline_name_or_path in WEIGHT_CONFIG_REGISTRY:
        return WEIGHT_CONFIG_REGISTRY[pipeline_name_or_path]

    # Try partial matches (for local paths that might include the weight ID)
    for registered_id, config_class in WEIGHT_CONFIG_REGISTRY.items():
        if registered_id in pipeline_name_or_path:
            return config_class

    # If no match, try to use the fallback config
    fallback_config = None
    # Try to determine pipeline architecture for fallback
    for pipeline_type, detector in PIPELINE_DETECTOR.items():
        if detector(pipeline_name.lower()):
            fallback_config = PIPELINE_FALLBACK_CONFIG.get(pipeline_type)
            break

    logger.warning("No match found for pipeline %s, using fallback config %s.",
                   pipeline_name_or_path, fallback_config)
    return fallback_config
