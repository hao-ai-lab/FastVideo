# SPDX-License-Identifier: Apache-2.0
import os
from typing import Any, cast

import torch

import fastvideo.envs as envs
from fastvideo.distributed import (cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel)
from fastvideo.distributed.parallel_state import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines import ForwardBatch, LoRAPipeline, build_pipeline

logger = init_logger(__name__)


def _log_cuda_device_uuid(rank: int, device: torch.device) -> None:
    """Record an NVIDIA worker UUID when external NVTX profiling is enabled."""
    if not envs.FASTVIDEO_NVTX_PROFILE.get():
        return
    device_uuid = torch.cuda.get_device_properties(device).uuid
    logger.info("Worker %d CUDA device UUID: GPU-%s", rank, device_uuid, local_main_process_only=False)


def _log_pipeline_memory(pipeline) -> None:
    """Debug (FASTVIDEO_MEMORY_REPORT=1): bytes held per pipeline component, by device and dtype, plus the
    largest tensors, so the resident footprint can be attributed before choosing offload placements."""
    gib = 1024**3
    for name, module in getattr(pipeline, "modules", {}).items():
        if not isinstance(module, torch.nn.Module):
            continue
        by_kind: dict[str, int] = {}
        largest: list[tuple[int, str, str]] = []
        seen: set[int] = set()
        for tname, t in list(module.named_parameters()) + list(module.named_buffers()):
            if t is None or id(t) in seen:
                continue
            seen.add(id(t))
            nbytes = t.numel() * t.element_size()
            key = f"{t.device.type}/{str(t.dtype).replace('torch.', '')}"
            by_kind[key] = by_kind.get(key, 0) + nbytes
            largest.append((nbytes, tname, key))
        largest.sort(reverse=True)
        total = sum(by_kind.values())
        logger.info("MEMREPORT %s total=%.2f GiB %s", name, total / gib, {
            k: round(v / gib, 2)
            for k, v in sorted(by_kind.items(), key=lambda kv: -kv[1])
        })
        for nbytes, tname, key in largest[:8]:
            logger.info("MEMREPORT %s   %.3f GiB %s %s", name, nbytes / gib, key, tname)
    if torch.cuda.is_available():
        logger.info("MEMREPORT cuda allocated=%.2f GiB reserved=%.2f GiB",
                    torch.cuda.memory_allocated() / gib,
                    torch.cuda.memory_reserved() / gib)


class Worker:

    def __init__(self, fastvideo_args: FastVideoArgs, local_rank: int, rank: int, distributed_init_method: str):
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
        envs.set_external("TORCH_NCCL_AVOID_RECORD_STREAMS", "1")
        # This env var set by Ray causes exceptions with graph building.
        envs.unset_external("NCCL_ASYNC_ERROR_HANDLING")

        # Set environment variables BEFORE calling get_local_torch_device()
        # so that each worker uses the correct device
        # Both multiprocessing and Ray pass the worker-local rank explicitly.
        # Ray deliberately excludes LOCAL_RANK from the copied driver
        # environment and exposes all GPUs assigned to the node, so leaving an
        # inherited or missing value here would bind every Ray actor to cuda:0.
        envs.set_external("LOCAL_RANK", str(self.local_rank))
        envs.set_external("RANK", str(self.rank))
        envs.set_external("WORLD_SIZE", str(self.fastvideo_args.num_gpus))

        # Platform-agnostic device initialization
        self.device = get_local_torch_device()

        from fastvideo.platforms import current_platform

        # Set the CUDA device BEFORE any CUDA calls
        if current_platform.is_cuda_alike():
            torch.cuda.set_device(self.device)
            # Debug: FASTVIDEO_CUDA_MEMORY_CAP_GIB emulates a smaller card by capping this process's allocator.
            cap_gib = os.environ.get("FASTVIDEO_CUDA_MEMORY_CAP_GIB")
            if cap_gib:
                total = torch.cuda.get_device_properties(self.device).total_memory
                torch.cuda.set_per_process_memory_fraction(min(1.0, float(cap_gib) * 1024**3 / total), self.device)
                logger.info("Capped CUDA allocator at %s GiB of %.1f GiB", cap_gib, total / 1024**3)
            self.init_gpu_memory = torch.cuda.mem_get_info(self.device)[0]
            if current_platform.is_cuda():
                _log_cuda_device_uuid(self.rank, self.device)
        else:
            # For MPS, we can't get memory info the same way
            self.init_gpu_memory = 0

        # CUDA's unified-memory classification reads runtime device
        # properties, so make this decision only after this worker has bound
        # its own device. The worker-local args object is what every loader and
        # pipeline stage below will consume.
        device_id = self.device.index if self.device.index is not None else 0
        self.fastvideo_args.finalize_device_offload_policy(device_id)

        # Initialize the distributed environment.
        maybe_init_distributed_environment_and_model_parallel(self.fastvideo_args.tp_size, self.fastvideo_args.sp_size,
                                                              self.distributed_init_method)

        self.pipeline = build_pipeline(self.fastvideo_args)
        if os.environ.get("FASTVIDEO_MEMORY_REPORT") == "1" and self.rank == 0:
            _log_pipeline_memory(self.pipeline)

    def execute_forward(self, forward_batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        output_batch = self.pipeline.forward(forward_batch, self.fastvideo_args)
        needs_output = forward_batch.return_frames or (forward_batch.save_video
                                                       and fastvideo_args.output_type != "latent"
                                                       and not output_batch.extra.get("audio_only"))
        if output_batch.output is not None and not needs_output:
            # Drop the decoded tensor before multiprocessing or Ray transports
            # the worker result back to the generator.
            output_batch.output = torch.empty(0, device="cpu")
        return cast(ForwardBatch, output_batch)

    def shutdown(self) -> dict[str, Any]:
        """Gracefully shut down the worker process"""
        logger.info("Worker %d shutting down...", self.rank, local_main_process_only=False)
        # Clean up resources
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            # Clean up pipeline resources if needed
            pass

        # Destroy the distributed environment
        cleanup_dist_env_and_memory(shutdown_ray=False)

        logger.info("Worker %d shutdown complete", self.rank, local_main_process_only=False)
        return {"status": "shutdown_complete"}

    def set_lora_adapter(self,
                         lora_nickname: str,
                         lora_path: str | None = None,
                         strength: float = 1.0,
                         accumulate: bool = False) -> dict[str, Any]:
        if isinstance(self.pipeline, LoRAPipeline):
            self.pipeline.set_lora_adapter(lora_nickname, lora_path, strength=strength, accumulate=accumulate)
            logger.info("Worker %d set LoRA adapter %s with path %s", self.rank, lora_nickname, lora_path)
            return {"status": "lora_adapter_set"}
        return {"status": "failed: pipeline is not a LoRAPipeline"}

    def unmerge_lora_weights(self) -> dict[str, Any]:
        if isinstance(self.pipeline, LoRAPipeline):
            self.pipeline.unmerge_lora_weights()
            return {"status": "lora_adapter_unmerged"}
        return {"status": "failed: pipeline is not a LoRAPipeline"}

    def execute_streaming_reset(self, forward_batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> dict[str, Any]:
        self.pipeline.streaming_reset(forward_batch, self.fastvideo_args)
        return {"status": "reset_complete"}

    def execute_streaming_step(self, keyboard_action: torch.Tensor, mouse_action: torch.Tensor) -> ForwardBatch:
        return self.pipeline.streaming_step(keyboard_action, mouse_action)

    def execute_streaming_clear(self) -> dict[str, Any]:
        self.pipeline.streaming_clear()
        return {"status": "cleared"}

    def merge_lora_weights(self) -> dict[str, Any]:
        if isinstance(self.pipeline, LoRAPipeline):
            self.pipeline.merge_lora_weights()
            return {"status": "lora_adapter_merged"}
        return {"status": "failed: pipeline is not a LoRAPipeline"}
