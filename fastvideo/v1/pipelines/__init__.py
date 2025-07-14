# SPDX-License-Identifier: Apache-2.0
"""
Diffusion pipelines for fastvideo.v1.

This package contains diffusion pipelines for generating videos and images.
"""

from typing import cast

from fastvideo.v1.fastvideo_args import FastVideoArgs, ExecutionMode
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.v1.pipelines.lora_pipeline import LoRAPipeline
from fastvideo.v1.pipelines.pipeline_batch_info import (ForwardBatch,
                                                        TrainingBatch)
from fastvideo.v1.pipelines.pipeline_registry import PipelineType, get_pipeline_registry
from fastvideo.v1.utils import (maybe_download_model,
                                verify_model_config_and_directory)

logger = init_logger(__name__)


class PipelineWithLoRA(LoRAPipeline, ComposedPipelineBase):
    """Type for a pipeline that has both ComposedPipelineBase and LoRAPipeline functionality."""
    pass


def build_pipeline(fastvideo_args: FastVideoArgs, pipeline_type: PipelineType | str = PipelineType.BASIC) -> PipelineWithLoRA:
    """
    Only works with valid hf diffusers configs. (model_index.json)
    We want to build a pipeline based on the inference args mode_path:
    1. download the model from the hub if it's not already downloaded
    2. verify the model config and directory
    3. based on the config, determine the pipeline class 
    """
    # Get pipeline type
    model_path = fastvideo_args.model_path
    model_path = maybe_download_model(model_path)
    # fastvideo_args.downloaded_model_path = model_path
    logger.info("Model path: %s", model_path)

    config = verify_model_config_and_directory(model_path)
    pipeline_name = config.get("_class_name")
    if pipeline_name is None:
        raise ValueError(
            "Model config does not contain a _class_name attribute. "
            "Only diffusers format is supported.")

    # Get the appropriate pipeline registry based on mode
    mode = fastvideo_args.mode
    logger.info("Building pipeline for mode: %s", mode.value if isinstance(mode, ExecutionMode) else mode)
    pipeline_registry = get_pipeline_registry(mode)

    if isinstance(pipeline_type, str):
        pipeline_type = PipelineType.from_string(pipeline_type)

    pipeline_cls = pipeline_registry.resolve_pipeline_cls(
        pipeline_name,
        pipeline_type,
        fastvideo_args.workload_type)

    # instantiate the pipelines
    pipeline = pipeline_cls(model_path, fastvideo_args)

    logger.info("Pipelines instantiated")

    return cast(PipelineWithLoRA, pipeline)


__all__ = [
    "build_pipeline",
    "ComposedPipelineBase",
    "ForwardBatch",
    "LoRAPipeline",
    "TrainingBatch",
]
