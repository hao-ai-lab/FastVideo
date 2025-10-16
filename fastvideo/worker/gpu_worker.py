# SPDX-License-Identifier: Apache-2.0
import os
from typing import Any, cast

import torch

from fastvideo.distributed import (
    cleanup_dist_env_and_memory,
    maybe_init_distributed_environment_and_model_parallel)
from fastvideo.distributed.parallel_state import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines import ForwardBatch, LoRAPipeline, build_pipeline

logger = init_logger(__name__)


class Worker:

    def __init__(self, fastvideo_args: FastVideoArgs, local_rank: int,
                 rank: int, distributed_init_method: str):
        self.fastvideo_args = fastvideo_args
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method

        # Init request dispatcher
        # TODO(will): add request dispatcher: use TypeBasedDispatcher from
        # utils.py
        # self._request_dispatcher = TypeBasedDispatcher(
        #     [
        # (RpcReqInput, self.handle_rpc_request),
        # (GenerateRequest, self.handle_generate_request),
        # (ExpertDistributionReq, self.expert_distribution_handle),
        #     ]
        # )

    def init_device(self) -> None:
        """Initialize the device for the worker."""

        # torch.distributed.all_reduce does not free the input tensor until
        # the synchronization point. This causes the memory usage to grow
        # as the number of all_reduce calls increases. This env var disables
        # this behavior.
        # Related issue:
        # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
        # This env var set by Ray causes exceptions with graph building.
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)

        # Platform-agnostic device initialization
        self.device = get_local_torch_device()

        from fastvideo.platforms import current_platform

        # _check_if_gpu_supports_dtype(self.model_config.dtype)
        if current_platform.is_cuda_alike():
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            # For MPS, we can't get memory info the same way
            self.init_gpu_memory = 0

        if self.fastvideo_args.distributed_executor_backend == "mp":
            os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.fastvideo_args.num_gpus)

        # Initialize the distributed environment.
        maybe_init_distributed_environment_and_model_parallel(
            self.fastvideo_args.tp_size, self.fastvideo_args.sp_size,
            self.distributed_init_method)

        self.pipeline = build_pipeline(self.fastvideo_args)

    def execute_forward(self, forward_batch: ForwardBatch,
                        fastvideo_args: FastVideoArgs) -> ForwardBatch:
        output_batch = self.pipeline.forward(forward_batch, self.fastvideo_args)
        return cast(ForwardBatch, output_batch)

    def shutdown(self) -> dict[str, Any]:
        """Gracefully shut down the worker process"""
        logger.info("Worker %d shutting down...",
                    self.rank,
                    local_main_process_only=False)
        # Clean up resources
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            # Clean up pipeline resources if needed
            pass

        # Destroy the distributed environment
        cleanup_dist_env_and_memory(shutdown_ray=False)

        logger.info("Worker %d shutdown complete",
                    self.rank,
                    local_main_process_only=False)
        return {"status": "shutdown_complete"}

    def set_lora_adapter(self,
                         lora_nickname: str,
                         lora_path: str | None = None) -> dict[str, Any]:
        if isinstance(self.pipeline, LoRAPipeline):
            self.pipeline.set_lora_adapter(lora_nickname, lora_path)
            logger.info("Worker %d set LoRA adapter %s with path %s", self.rank,
                        lora_nickname, lora_path)
            return {"status": "lora_adapter_set"}
        return {"status": "failed: pipeline is not a LoRAPipeline"}

    def unmerge_lora_weights(self) -> dict[str, Any]:
        if isinstance(self.pipeline, LoRAPipeline):
            self.pipeline.unmerge_lora_weights()
            return {"status": "lora_adapter_unmerged"}
        return {"status": "failed: pipeline is not a LoRAPipeline"}

    def merge_lora_weights(self) -> dict[str, Any]:
        if isinstance(self.pipeline, LoRAPipeline):
            self.pipeline.merge_lora_weights()
            return {"status": "lora_adapter_merged"}
        return {"status": "failed: pipeline is not a LoRAPipeline"}
