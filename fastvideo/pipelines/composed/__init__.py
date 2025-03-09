"""
Composed pipelines for diffusion models.

This package contains pipelines that are composed of multiple stages.
"""

from fastvideo.pipelines.composed.composed_pipeline_base import (
    ComposedPipelineBase,
    DiffusionPipelineOutput,
)

__all__ = [
    "ComposedPipelineBase",
    "DiffusionPipelineOutput",
]

# Note: Do not import TextToVideoPipeline here to avoid circular imports.
# It will be imported in the main pipelines/__init__.py file. 