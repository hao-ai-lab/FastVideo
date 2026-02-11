# SPDX-License-Identifier: Apache-2.0
"""
Diffusion pipelines for fastvideo.

This package contains diffusion pipelines for generating videos and images.
"""

from typing import cast

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.lora_pipeline import LoRAPipeline
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch, TrainingBatch
from fastvideo.pipelines.pipeline_registry import PipelineType
from fastvideo.registry import get_model_info
from fastvideo.utils import maybe_download_model

logger = init_logger(__name__)


class PipelineWithLoRA(LoRAPipeline, ComposedPipelineBase):
    """Type for a pipeline that has both ComposedPipelineBase and LoRAPipeline functionality."""
    pass


def build_pipeline(
        fastvideo_args: FastVideoArgs,
        pipeline_type: PipelineType | str = PipelineType.BASIC
) -> PipelineWithLoRA:
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

    logger.info(
        "Building pipeline of type: %s", pipeline_type.value if isinstance(
            pipeline_type, PipelineType) else pipeline_type)

    model_info = get_model_info(
        model_path=model_path,
        pipeline_type=pipeline_type,
        workload_type=fastvideo_args.workload_type,
        override_pipeline_cls_name=fastvideo_args.override_pipeline_cls_name,
    )
    pipeline_cls = model_info.pipeline_cls

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
