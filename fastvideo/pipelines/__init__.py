"""
Diffusion pipelines for FastVideo.

This package contains diffusion pipelines for generating videos and images.
"""

from typing import Dict, Optional, Type, Any

# First, import the registry
from fastvideo.pipelines.pipeline_registry import PipelineRegistry, register_pipeline

# Then import the base classes
from fastvideo.pipelines.composed.composed_pipeline_base import (
    ComposedPipelineBase, 
    DiffusionPipelineOutput
)

# Function to create a pipeline
def create_pipeline(
    pipeline_type: str,
    **kwargs
) -> ComposedPipelineBase:
    """
    Create a pipeline of the specified type.
    
    Args:
        pipeline_type: The type of pipeline to create.
        **kwargs: Additional arguments to pass to the pipeline constructor.
        
    Returns:
        The created pipeline.
        
    Raises:
        ValueError: If the pipeline type is not recognized.
    """
    pipeline_cls = PipelineRegistry.get(pipeline_type)
    if pipeline_cls is None:
        available_pipelines = list(PipelineRegistry.list().keys())
        raise ValueError(
            f"Pipeline type '{pipeline_type}' not recognized. "
            f"Available pipeline types: {available_pipelines}"
        )
    
    return pipeline_cls(**kwargs)


def list_available_pipelines() -> Dict[str, Type[Any]]:
    """
    List all available pipeline types.
    
    Returns:
        A dictionary of pipeline names to pipeline classes.
    """
    return PipelineRegistry.list()


# Import all pipeline implementations to register them
# These imports should be at the end to avoid circular imports
from fastvideo.pipelines.composed.text_to_video import TextToVideoPipeline
from fastvideo.pipelines.implementations.hunyuan import HunYuanVideoPipeline

__all__ = [
    "create_pipeline",
    "list_available_pipelines",
    "ComposedPipelineBase",
    "DiffusionPipelineOutput",
    "register_pipeline",
    "TextToVideoPipeline",
    "HunYuanVideoPipeline",
] 