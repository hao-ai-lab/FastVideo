# SPDX-License-Identifier: Apache-2.0
"""
Post-processing stage for diffusion pipelines.
"""

from typing import Union

import numpy as np
import torch
from diffusers.utils import BaseOutput

from fastvideo.v1.inference_args import InferenceArgs
from fastvideo.v1.logger import init_logger

from ..pipeline_batch_info import ForwardBatch
from .base import PipelineStage

logger = init_logger(__name__)


class PostProcessingStage(PipelineStage):
    """
    Stage for post-processing the decoded outputs.
    
    This stage handles any final processing needed on the decoded outputs,
    such as format conversion, normalization, etc.
    """

    def forward(
        self,
        batch: ForwardBatch,
        inference_args: InferenceArgs,
    ) -> ForwardBatch:
        """
        Apply post-processing to the results.
        
        Args:
            batch: The current batch information.
            inference_args: The inference arguments.
            
        Returns:
            The batch with post-processed outputs.
        """
        videos = batch.videos

        # Convert to numpy if requested
        if inference_args.output_type == "numpy":
            videos = videos.numpy()

        # Create output object
        output = DiffusionPipelineOutput(videos=videos)
        batch.output = output

        return batch


class DiffusionPipelineOutput(BaseOutput):
    """
    Output class for diffusion pipelines.
    
    Args:
        videos: The generated videos.
    """
    videos: Union[torch.Tensor, np.ndarray]
