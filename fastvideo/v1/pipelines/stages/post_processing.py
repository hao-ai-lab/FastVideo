# SPDX-License-Identifier: Apache-2.0

"""
Post-processing stage for diffusion pipelines.
"""

import torch
import numpy as np
from typing import Optional, Union, List, Tuple

from fastvideo.v1.pipelines.stages.base import PipelineStage
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.inference_args import InferenceArgs
from diffusers.utils import BaseOutput
from fastvideo.v1.logger import init_logger

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
