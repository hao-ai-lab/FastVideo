"""
Base classes for pipeline stages.

This module defines the abstract base classes for pipeline stages that can be
composed to create complete diffusion pipelines.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List, Tuple
import torch

from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.inference_args import InferenceArgs


class PipelineStage(ABC):
    """
    Abstract base class for all pipeline stages.
    
    A pipeline stage represents a discrete step in the diffusion process that can be
    composed with other stages to create a complete pipeline. Each stage is responsible
    for a specific part of the process, such as prompt encoding, latent preparation, etc.
    """
    
    def __init__(self):
        """Initialize the pipeline stage."""
        pass
    
    @property
    def device(self) -> torch.device:
        """Get the device for this stage."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def __call__(
        self,
        batch: ForwardBatch,
        inference_args: InferenceArgs,
    ) -> ForwardBatch:
        """
        Execute the stage's processing on the batch.
        
        Args:
            batch: The current batch information.
            inference_args: The inference arguments.
            
        Returns:
            The updated batch information after this stage's processing.
        """
        pass
    
    def register_modules(self, **kwargs):
        """
        Register modules needed by this stage.
        
        Args:
            **kwargs: The modules to register.
        """
        for name, module in kwargs.items():
            setattr(self, name, module)


class InputValidationStage(PipelineStage):
    """Base class for input validation stages."""
    
    @abstractmethod
    def __call__(
        self,
        batch: ForwardBatch,
        inference_args: InferenceArgs,
    ) -> ForwardBatch:
        """Validate and prepare inputs."""
        pass


class PromptEncodingStage(PipelineStage):
    """Base class for prompt encoding stages."""
    
    @abstractmethod
    def __call__(
        self,
        batch: ForwardBatch,
        inference_args: InferenceArgs,
    ) -> ForwardBatch:
        """Encode the prompt(s)."""
        pass


class TimestepPreparationStage(PipelineStage):
    """Base class for timestep preparation stages."""
    
    @abstractmethod
    def __call__(
        self,
        batch: ForwardBatch,
        inference_args: InferenceArgs,
    ) -> ForwardBatch:
        """Prepare timesteps for the diffusion process."""
        pass


class LatentPreparationStage(PipelineStage):
    """Base class for latent preparation stages."""
    
    @abstractmethod
    def __call__(
        self,
        batch: ForwardBatch,
        inference_args: InferenceArgs,
    ) -> ForwardBatch:
        """Prepare initial latent variables."""
        pass


class ConditioningStage(PipelineStage):
    """Base class for conditioning stages (e.g., classifier-free guidance)."""
    
    @abstractmethod
    def __call__(
        self,
        batch: ForwardBatch,
        inference_args: InferenceArgs,
    ) -> ForwardBatch:
        """Apply conditioning to the model inputs."""
        pass


class DenoisingStage(PipelineStage):
    """Base class for the denoising loop stage."""
    
    @abstractmethod
    def __call__(
        self,
        batch: ForwardBatch,
        inference_args: InferenceArgs,
    ) -> ForwardBatch:
        """Run the denoising loop."""
        pass


class DecodingStage(PipelineStage):
    """Base class for decoding stages."""
    
    @abstractmethod
    def __call__(
        self,
        batch: ForwardBatch,
        inference_args: InferenceArgs,
    ) -> ForwardBatch:
        """Decode the results."""
        pass


class PostProcessingStage(PipelineStage):
    """Base class for post-processing stages."""
    
    @abstractmethod
    def __call__(
        self,
        batch: ForwardBatch,
        inference_args: InferenceArgs,
    ) -> ForwardBatch:
        """Apply post-processing to the results."""
        pass 