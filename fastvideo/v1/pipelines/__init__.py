# SPDX-License-Identifier: Apache-2.0

"""
Diffusion pipelines for fastvideo.v1.

This package contains diffusion pipelines for generating videos and images.
"""
from typing import Dict, Type, Any

from fastvideo.v1.pipelines.pipeline_registry import PipelineRegistry
from fastvideo.v1.inference_args import InferenceArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.composed_pipeline_base import (
    ComposedPipelineBase)

from fastvideo.v1.utils import maybe_download_model, verify_model_config_and_directory

logger = init_logger(__name__)


def build_pipeline(inference_args: InferenceArgs) -> ComposedPipelineBase:
    """
    Only works with valid hf diffusers configs. (model_index.json)
    We want to build a pipeline based on the inference args mode_path:
    1. download the model from the hub if it's not already downloaded
    2. verify the model config and directory
    3. based on the config, determine the pipeline class 
    """
    # Get pipeline type
    model_path = inference_args.model_path
    model_path = maybe_download_model(model_path)
    # inference_args.downloaded_model_path = model_path
    logger.info(f"Model path: {model_path}")
    config = verify_model_config_and_directory(model_path)

    pipeline_architecture = config.get("_class_name")
    if pipeline_architecture is None:
        raise ValueError(
            "Model config does not contain a _class_name attribute. "
            "Only diffusers format is supported.")

    pipeline_cls, pipeline_architecture = PipelineRegistry.resolve_pipeline_cls(
        pipeline_architecture)

    # instantiate the pipeline
    pipeline = pipeline_cls(model_path, inference_args)
    logger.info(f"Pipeline instantiated")

    # pipeline is now initialized and ready to use
    return pipeline


def list_available_pipelines() -> Dict[str, Type[Any]]:
    """
    List all available pipeline types.
    
    Returns:
        A dictionary of pipeline names to pipeline classes.
    """
    return PipelineRegistry.list()


__all__ = [
    "build_pipeline",
    "list_available_pipelines",
    "ComposedPipelineBase",]
