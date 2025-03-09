"""
Diffusion pipelines for FastVideo.

This package contains diffusion pipelines for generating videos and images.
"""

from typing import Dict, Optional, Type, Any

# First, import the registry
from fastvideo.pipelines.pipeline_registry import PipelineRegistry, register_pipeline
from fastvideo.inference_args import InferenceArgs

# Then import the base classes
from fastvideo.pipelines.composed.composed_pipeline_base import (
    ComposedPipelineBase, 
    DiffusionPipelineOutput
)

def get_pipeline_type(inference_args: InferenceArgs) -> str:
    # hardcode for now
    return "hunyuan_video"

def build_pipeline(inference_args: InferenceArgs) -> ComposedPipelineBase:
    """
    Only works with valid hf diffusers configs. (model_index.json)
    We want to build a pipeline based on the inference args.
    1. load the correct hf config from disk or download from hub
    2. based on the config, determine the pipeline class and pipeline loader class
    3. parse the config to get the model component (vae, text_encoders, etc...)
    names and paths
    4. the pipeline loader class will use the model component names and paths to load
    the models. 
    5. the pipeline class will be composed of the models returned by the pipeline loader
    """
    # Get pipeline type
    pipeline_type = get_pipeline_type(inference_args)
    
    if not hasattr(inference_args, 'model_type') or not inference_args.model_type:
        inference_args.model_type = pipeline_type
    
    # TODO(will): Get pipeline loader (hardcoded for now)
    from fastvideo.pipelines.loader import get_pipeline_loader
    pipeline_loader = get_pipeline_loader(inference_args)
    
    # Load and return pipeline
    return pipeline_loader.load_pipeline_v2(inference_args)


def list_available_pipelines() -> Dict[str, Type[Any]]:
    """
    List all available pipeline types.
    
    Returns:
        A dictionary of pipeline names to pipeline classes.
    """
    return PipelineRegistry.list()


# Import all pipeline implementations to register them
# These imports should be at the end to avoid circular imports
from fastvideo.pipelines.implementations.hunyuan import HunyuanVideoPipeline

__all__ = [
    "create_pipeline",
    "list_available_pipelines",
    "ComposedPipelineBase",
    "DiffusionPipelineOutput",
    "register_pipeline",
    "HunyuanVideoPipeline",
] 