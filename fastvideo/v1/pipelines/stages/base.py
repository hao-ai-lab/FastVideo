# SPDX-License-Identifier: Apache-2.0

"""
Base classes for pipeline stages.

This module defines the abstract base classes for pipeline stages that can be
composed to create complete diffusion pipelines.
"""

from abc import ABC, abstractmethod
import torch
import time
import traceback

from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.inference_args import InferenceArgs
from fastvideo.v1.logger import init_logger

logger = init_logger(__name__)


class PipelineStage(ABC):
    """
    Abstract base class for all pipeline stages.
    
    A pipeline stage represents a discrete step in the diffusion process that can be
    composed with other stages to create a complete pipeline. Each stage is responsible
    for a specific part of the process, such as prompt encoding, latent preparation, etc.
    """

    def __init__(self):
        pass

    @property
    def device(self) -> torch.device:
        """Get the device for this stage."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_logging(self, enable: bool):
        """
        Enable or disable logging for this stage.
        
        Args:
            enable: Whether to enable logging.
        """
        self._enable_logging = enable

    def __call__(
        self,
        batch: ForwardBatch,
        inference_args: InferenceArgs,
    ) -> ForwardBatch:
        """
        Execute the stage's processing on the batch with optional logging.
        Should not be overridden by subclasses.
        
        Args:
            batch: The current batch information.
            inference_args: The inference arguments.
            
        Returns:
            The updated batch information after this stage's processing.
        """
        # if envs.ENABLE_STAGE_LOGGING:
        if False:
            self._logger.info(f"[{self._stage_name}] Starting execution")
            start_time = time.time()

            try:
                # Call the actual implementation
                result = self._call_implementation(batch, inference_args)

                execution_time = time.time() - start_time
                self._logger.info(
                    f"[{self._stage_name}] Execution completed in {execution_time * 1000:.2f} ms"
                )

                return result
            except Exception as e:
                execution_time = time.time() - start_time
                self._logger.error(
                    f"[{self._stage_name}] Error during execution after {execution_time * 1000:.2f} ms: {e}"
                )
                self._logger.error(
                    f"[{self._stage_name}] Traceback: {traceback.format_exc()}")

                # Re-raise the exception
                raise
        else:
            # Just call the implementation directly if logging is disabled
            # TODO(will): Also handle backward
            return self.forward(batch, inference_args)

    @abstractmethod
    def forward(
        self,
        batch: ForwardBatch,
        inference_args: InferenceArgs,
    ) -> ForwardBatch:
        """
        Forward pass of the stage's processing.
        
        This method should be implemented by subclasses to provide the forward
        processing logic for the stage.
        
        Args:
            batch: The current batch information.
            inference_args: The inference arguments.
            
        Returns:
            The updated batch information after this stage's processing.
        """
        raise NotImplementedError

    def backward(
        self,
        batch: ForwardBatch,
        inference_args: InferenceArgs,
    ) -> ForwardBatch:
        raise NotImplementedError
