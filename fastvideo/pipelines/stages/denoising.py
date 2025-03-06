"""
Denoising stage for diffusion pipelines.
"""

from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.inference_args import InferenceArgs


class DenoisingStage(PipelineStage):
    """Base class for the denoising loop stage."""
    
    def _call_implementation(
        self,
        batch: ForwardBatch,
        inference_args: InferenceArgs,
    ) -> ForwardBatch:
        """Run the denoising loop."""
        pass 