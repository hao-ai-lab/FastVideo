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
from fastvideo.v1.pipelines.stages.validators import VerificationResult

logger = init_logger(__name__)


class StageVerificationError(Exception):
    """Exception raised when stage verification fails."""
    pass


class PipelineStage(ABC):
    """
    Abstract base class for all pipeline stages.
    
    A pipeline stage represents a discrete step in the diffusion process that can be
    composed with other stages to create a complete pipeline. Each stage is responsible
    for a specific part of the process, such as prompt encoding, latent preparation, etc.
    """

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        """
        Verify the input for the stage.

        Example:
            from fastvideo.v1.pipelines.stages.validators import V, VerificationResult
            
            def verify_input(self, batch, fastvideo_args):
                result = VerificationResult()
                result.add_check("height", batch.height, V.positive_int_divisible(8))
                result.add_check("width", batch.width, V.positive_int_divisible(8))
                result.add_check("image_latent", batch.image_latent, V.is_tensor)
                return result

        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.

        Returns:
            A VerificationResult containing the verification status.
        
        """
        # Default implementation - no verification
        return VerificationResult()

    def verify_output(self, batch: ForwardBatch,
                      fastvideo_args: FastVideoArgs) -> VerificationResult:
        """
        Verify the output for the stage.

        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.

        Returns:
            A VerificationResult containing the verification status.
        """
        # Default implementation - no verification
        return VerificationResult()

    def _run_verification(self, verification_result: VerificationResult,
                          stage_name: str, verification_type: str) -> None:
        """
        Run verification and raise errors if any checks fail.
        
        Args:
            verification_result: Results from verify_input or verify_output
            stage_name: Name of the current stage
            verification_type: "input" or "output"
        """
        if not verification_result.is_valid():
            failed_fields = verification_result.get_failed_fields()
            if failed_fields:
                # Get detailed failure information
                detailed_summary = verification_result.get_failure_summary()

                failed_fields_str = ", ".join(failed_fields)
                error_msg = (
                    f"{verification_type.capitalize()} verification failed for {stage_name}: "
                    f"Failed fields: {failed_fields_str}\n"
                    f"Details: {detailed_summary}")
                raise StageVerificationError(error_msg)

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
        Execute the stage's processing on the batch with optional verification and logging.
        Should not be overridden by subclasses.
        
        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.
            
        Returns:
            The updated batch information after this stage's processing.
        """
        stage_name = self.__class__.__name__

        # Check if verification is enabled (simple approach for prototype)
        enable_verification = getattr(fastvideo_args,
                                      'enable_stage_verification', False)

        if enable_verification:
            # Pre-execution input verification
            try:
                input_result = self.verify_input(batch, fastvideo_args)
                self._run_verification(input_result, stage_name, "input")
            except Exception as e:
                logger.error("Input verification failed for %s: %s", stage_name,
                             str(e))
                raise

        # Execute the actual stage logic
        # envs.ENABLE_STAGE_LOGGING
        if False:
            self._logger.info("[%s] Starting execution", stage_name)
            start_time = time.perf_counter()

            try:
                result = self.forward(batch, fastvideo_args)
                execution_time = time.perf_counter() - start_time
                self._logger.info("[%s] Execution completed in %s ms",
                                  stage_name, execution_time * 1000)
            except Exception as e:
                execution_time = time.perf_counter() - start_time
                self._logger.error(
                    "[%s] Error during execution after %s ms: %s", stage_name,
                    execution_time * 1000, e)
                self._logger.error("[%s] Traceback: %s", stage_name,
                                   traceback.format_exc())
                raise
        else:
            # Direct execution (current behavior)
            result = self.forward(batch, fastvideo_args)

        if enable_verification:
            # Post-execution output verification
            try:
                output_result = self.verify_output(result, fastvideo_args)
                self._run_verification(output_result, stage_name, "output")
            except Exception as e:
                logger.error("Output verification failed for %s: %s",
                             stage_name, str(e))
                raise

        return result

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
