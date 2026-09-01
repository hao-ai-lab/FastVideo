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

import fastvideo.envs as envs
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.lazy_module import LazyModule
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.validators import VerificationResult

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
    performance_component_metric: str | None = None
    # Deferred modules this stage is the last user of, installed by the
    # pipeline under ``lazy_module_load``. Released once __call__ returns.
    # Living here rather than in the pipeline's stage loop means a pipeline
    # that overrides forward still frees, since __call__ is the one entry
    # point subclasses are told not to override.
    _lazy_modules_to_release: tuple[LazyModule, ...] = ()

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        """
        Verify the input for the stage.

        Example:
            from fastvideo.pipelines.stages.validators import V, VerificationResult
            
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

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
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

    def _run_verification(self, verification_result: VerificationResult, stage_name: str,
                          verification_type: str) -> None:
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
                error_msg = (f"{verification_type.capitalize()} verification failed for {stage_name}: "
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
        stage_class_name = self.__class__.__name__
        stage_key = getattr(self, "_pipeline_stage_name", stage_class_name)
        stage_name = f"{stage_key}|{stage_class_name}"

        # Check if verification is enabled (simple approach for prototype)
        enable_verification = getattr(fastvideo_args, 'enable_stage_verification', False)

        if enable_verification:
            # Pre-execution input verification
            try:
                input_result = self.verify_input(batch, fastvideo_args)
                self._run_verification(input_result, stage_name, "input")
            except Exception as e:
                logger.error("Input verification failed for %s: %s", stage_name, str(e))
                raise

        # Execute the actual stage logic, then optional output verification.
        # One BaseException net: KeyboardInterrupt inside verify_output must
        # still free this stage's deferred modules (OOM is already an Exception).
        try:
            result = self._execute(batch, fastvideo_args, stage_key, stage_class_name, stage_name)
            if enable_verification:
                try:
                    output_result = self.verify_output(result, fastvideo_args)
                    self._run_verification(output_result, stage_name, "output")
                except Exception as e:
                    logger.error("Output verification failed for %s: %s", stage_name, str(e))
                    raise
        except BaseException:
            self._release_deferred_modules(stage_name)
            raise

        self._release_deferred_modules(stage_name)
        return result

    def _execute(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
        stage_key: str,
        stage_class_name: str,
        stage_name: str,
    ) -> ForwardBatch:
        """Run forward, with the optional timing and logging wrapper."""
        if envs.FASTVIDEO_STAGE_LOGGING:
            logger.info("[%s] Starting execution", stage_name)
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            try:
                result = self.forward(batch, fastvideo_args)
                torch.cuda.synchronize()
                execution_time = time.perf_counter() - start_time
                logger.info("[%s] Execution completed in %s ms", stage_name, execution_time * 1000)
                batch.logging_info.add_stage_execution_time(stage_key, execution_time)
                batch.logging_info.add_stage_metric(stage_key, "stage_class", stage_class_name)
                component_metric = self.performance_component_metric
                if component_metric is not None:
                    batch.logging_info.add_stage_metric(stage_key, "component_metric", component_metric)
            except Exception as e:
                torch.cuda.synchronize()
                execution_time = time.perf_counter() - start_time
                logger.error("[%s] Error during execution after %s ms: %s", stage_name, execution_time * 1000, e)
                logger.error("[%s] Traceback: %s", stage_name, traceback.format_exc())
                raise
        else:
            # Direct execution (current behavior)
            result = self.forward(batch, fastvideo_args)

        return result

    def _release_deferred_modules(self, stage_name: str) -> None:
        """Free the deferred components this stage is the last user of.

        Called on the way out whether or not the stage succeeded. A stage that
        raises after materializing a multi-gigabyte component would otherwise
        keep it for the life of the generator, and the retry that a
        memory-constrained caller is most likely to attempt would start from a
        worse position than the request that just failed.
        """
        for lazy_module in self._lazy_modules_to_release:
            try:
                lazy_module.release()
            except Exception:
                # Never let cleanup replace the exception being propagated.
                logger.exception("Failed to release deferred module after %s", stage_name)

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
