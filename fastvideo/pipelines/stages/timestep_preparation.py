"""
Timestep preparation stages for diffusion pipelines.

This module contains implementations of timestep preparation stages for diffusion pipelines.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import torch

from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.inference_args import InferenceArgs
from fastvideo.pipelines.stages.base import TimestepPreparationStage


class StandardTimestepPreparationStage(TimestepPreparationStage):
    """
    Standard timestep preparation stage for diffusion pipelines.
    
    This stage prepares the timesteps for the diffusion process.
    """
    
    needs_scheduler = True
    
    def __init__(self):
        """Initialize the timestep preparation stage."""
        super().__init__()
    
    def __call__(
        self,
        batch: ForwardBatch,
        inference_args: InferenceArgs,
    ) -> ForwardBatch:
        """
        Prepare timesteps for the diffusion process.
        
        Args:
            batch: The current batch information.
            inference_args: The inference arguments.
            
        Returns:
            The updated batch information after timestep preparation.
        """
        # Get the number of inference steps
        num_inference_steps = batch.num_inference_steps
        
        # Set the scheduler's timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        
        # Get the timesteps
        timesteps = self.scheduler.timesteps
        
        # Update the batch
        batch.timesteps = timesteps
        
        return batch


class FlowMatchingTimestepPreparationStage(TimestepPreparationStage):
    """
    Timestep preparation stage for flow matching diffusion pipelines.
    
    This stage prepares the timesteps for flow matching diffusion processes.
    """
    
    needs_scheduler = True
    
    def __init__(self, flow_shift: int = 7, flow_reverse: bool = False):
        """
        Initialize the flow matching timestep preparation stage.
        
        Args:
            flow_shift: The flow shift parameter.
            flow_reverse: Whether to reverse the flow direction.
        """
        super().__init__()
        self.flow_shift = flow_shift
        self.flow_reverse = flow_reverse
    
    def __call__(
        self,
        batch: ForwardBatch,
        inference_args: InferenceArgs,
    ) -> ForwardBatch:
        """
        Prepare timesteps for the flow matching diffusion process.
        
        Args:
            batch: The current batch information.
            inference_args: The inference arguments.
            
        Returns:
            The updated batch information after timestep preparation.
        """
        # Get the number of inference steps
        num_inference_steps = batch.num_inference_steps
        
        # Configure the scheduler
        if hasattr(self.scheduler, "set_flow_parameters"):
            self.scheduler.set_flow_parameters(
                shift=inference_args.flow_shift if inference_args.flow_shift is not None else self.flow_shift,
                reverse=inference_args.flow_reverse if hasattr(inference_args, "flow_reverse") else self.flow_reverse,
            )
        
        # Set the scheduler's timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        
        # Get the timesteps
        timesteps = self.scheduler.timesteps
        
        # Update the batch
        batch.timesteps = timesteps
        
        return batch 