# SPDX-License-Identifier: Apache-2.0
"""
Base classes for pipeline stages.

This module defines the abstract base classes for pipeline stages that can be
composed to create complete diffusion pipelines.
"""

import time
import traceback
from abc import ABC, abstractmethod

import torch

from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch

logger = init_logger(__name__)


class PipelineStage(ABC):
    """
    Abstract base class for all pipeline stages.
    
    A pipeline stage represents a discrete step in the diffusion process that can be
    composed with other stages to create a complete pipeline. Each stage is responsible
    for a specific part of the process, such as prompt encoding, latent preparation, etc.
    """

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
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Execute the stage's processing on the batch with optional logging.
        Should not be overridden by subclasses.
        
        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.
            
        Returns:
            The updated batch information after this stage's processing.
        """
        # if envs.ENABLE_STAGE_LOGGING:
        if False:
            self._logger.info("[%s] Starting execution", self._stage_name)
            start_time = time.perf_counter()

            try:
                # Call the actual implementation
                result = self._call_implementation(batch, fastvideo_args)

                execution_time = time.perf_counter() - start_time
                self._logger.info("[%s] Execution completed in %s ms",
                                  self._stage_name, execution_time * 1000)

                return result
            except Exception as e:
                execution_time = time.perf_counter() - start_time
                self._logger.error(
                    "[%s] Error during execution after %s ms: %s",
                    self._stage_name, execution_time * 1000, e)
                self._logger.error("[%s] Traceback: %s", self._stage_name,
                                   traceback.format_exc())

                # Re-raise the exception
                raise
        else:
            # Just call the implementation directly if logging is disabled
            # TODO(will): Also handle backward
            return self.forward(batch, fastvideo_args)

    @abstractmethod
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Forward pass of the stage's processing.
        
        This method should be implemented by subclasses to provide the forward
        processing logic for the stage.
        
        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.
            
        Returns:
            The updated batch information after this stage's processing.
        """
        raise NotImplementedError

    def backward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        raise NotImplementedError
