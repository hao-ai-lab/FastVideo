"""
Conditioning stage for diffusion pipelines.
"""

from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.inference_args import InferenceArgs


class ConditioningStage(PipelineStage):
    """Base class for conditioning stages (e.g., classifier-free guidance)."""
    
    def _call_implementation(
        self,
        batch: ForwardBatch,
        inference_args: InferenceArgs,
    ) -> ForwardBatch:
        """Apply conditioning to the model inputs."""
        pass 