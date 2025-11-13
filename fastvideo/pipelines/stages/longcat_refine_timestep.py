# SPDX-License-Identifier: Apache-2.0
"""
LongCat refinement timestep preparation stage.

This stage prepares special timesteps for LongCat refinement that start from t_thresh.
"""

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage

logger = init_logger(__name__)


class LongCatRefineTimestepStage(PipelineStage):
    """
    Stage for preparing timesteps specific to LongCat refinement.
    
    For refinement, we need to start from t_thresh instead of t=1.0, so we:
    1. Generate normal timesteps for num_inference_steps
    2. Filter to only keep timesteps < t_thresh * 1000
    3. Prepend t_thresh * 1000 as the first timestep
    """

    def __init__(self, scheduler) -> None:
        super().__init__()
        self.scheduler = scheduler

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Prepare refinement-specific timesteps.
        
        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.
            
        Returns:
            The batch with refinement timesteps.
        """
        # Only apply if this is a refinement task
        if batch.refine_from is None:
            return batch
        
        device = get_local_torch_device()
        num_inference_steps = batch.num_inference_steps
        t_thresh = batch.t_thresh
        
        logger.info(f"Preparing LongCat refinement timesteps (t_thresh={t_thresh})")
        
        # Generate sigmas for normal schedule
        # For FlowMatchEulerDiscreteScheduler, we need to construct sigmas
        # that match the refinement starting point
        
        # First set timesteps normally to get the base schedule
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        base_timesteps = self.scheduler.timesteps
        
        # Filter timesteps to only those < t_thresh * 1000
        t_thresh_value = t_thresh * 1000
        t_thresh_tensor = torch.tensor(t_thresh_value, dtype=base_timesteps.dtype, device=device)
        filtered_timesteps = base_timesteps[base_timesteps < t_thresh_tensor]
        
        # Prepend t_thresh as the starting timestep
        timesteps = torch.cat([t_thresh_tensor.unsqueeze(0), filtered_timesteps])
        
        # Update scheduler with these custom timesteps
        self.scheduler.timesteps = timesteps
        
        # Reconstruct sigmas: sigma = timestep / 1000, plus a trailing zero
        sigmas = torch.cat([timesteps / 1000.0, torch.zeros(1, device=device)])
        self.scheduler.sigmas = sigmas
        
        logger.info(f"Refinement timesteps: {len(timesteps)} steps starting from t={t_thresh}")
        logger.info(f"First few timesteps: {timesteps[:5].tolist()}")
        
        # Store in batch
        batch.timesteps = timesteps
        
        return batch

