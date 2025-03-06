"""
Post-processing stage for diffusion pipelines.
"""

from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.inference_args import InferenceArgs


class PostProcessingStage(PipelineStage):
    """Base class for post-processing stages."""
    
    def _call_implementation(
        self,
        batch: ForwardBatch,
        inference_args: InferenceArgs,
    ) -> ForwardBatch:
        """Apply post-processing to the results."""
        pass 