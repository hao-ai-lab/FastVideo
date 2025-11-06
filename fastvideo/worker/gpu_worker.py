# SPDX-License-Identifier: Apache-2.0
import contextlib
import faulthandler
import multiprocessing as mp
import os
import signal
import sys
from multiprocessing.connection import Connection
from typing import Any, TextIO, cast

import psutil
import torch

import fastvideo.envs as envs
from fastvideo.distributed import (
    cleanup_dist_env_and_memory,
    maybe_init_distributed_environment_and_model_parallel)
from fastvideo.distributed.parallel_state import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines import ForwardBatch, LoRAPipeline, build_pipeline
from fastvideo.platforms import current_platform
from fastvideo.utils import (get_exception_traceback,
                             kill_itself_when_parent_died)

logger = init_logger(__name__)

# ANSI color codes
CYAN = '\033[1;36m'
RESET = '\033[0;0m'


class Worker:

    def __init__(self, fastvideo_args: FastVideoArgs, local_rank: int,
                 rank: int, pipe: Connection, master_port: int):
        self.fastvideo_args = fastvideo_args
        self.local_rank = local_rank
        self.rank = rank
        # TODO(will): don't hardcode this
        self.distributed_init_method = "env://"
        self.pipe = pipe
        self.master_port = master_port
        self.init_device()

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

        # _check_if_gpu_supports_dtype(self.model_config.dtype)
        if current_platform.is_cuda_alike():
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            # For MPS, we can't get memory info the same way
            self.init_gpu_memory = 0

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.fastvideo_args.num_gpus)

        # Initialize the distributed environment.
        maybe_init_distributed_environment_and_model_parallel(
            self.fastvideo_args.tp_size, self.fastvideo_args.sp_size)

        self.pipeline = build_pipeline(self.fastvideo_args)

    def execute_forward(self, forward_batch: ForwardBatch,
                        fastvideo_args: FastVideoArgs) -> ForwardBatch:
        output_batch = self.pipeline.forward(forward_batch, self.fastvideo_args)
        return cast(ForwardBatch, output_batch)

    def set_lora_adapter(self,
                         lora_nickname: str,
                         lora_path: str | None = None) -> None:
        self.pipeline.set_lora_adapter(lora_nickname, lora_path)

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

    def unmerge_lora_weights(self) -> None:
        if isinstance(self.pipeline, LoRAPipeline):
            self.pipeline.unmerge_lora_weights()

    def merge_lora_weights(self) -> None:
        if isinstance(self.pipeline, LoRAPipeline):
            self.pipeline.merge_lora_weights()

    def event_loop(self) -> None:
        """Event loop for the worker."""
        logger.info("Worker %d starting event loop...",
                    self.rank,
                    local_main_process_only=False)
        while True:
            try:
                recv_rpc = self.pipe.recv()
                method_name = recv_rpc.get('method')

                # Handle shutdown request
                if method_name == 'shutdown':
                    response = self.shutdown()
                    with contextlib.suppress(Exception):
                        self.pipe.send(response)
                    break  # Exit the loop

                # Handle regular RPC calls
                if method_name == 'execute_forward':
                    forward_batch = recv_rpc['kwargs']['forward_batch']
                    fastvideo_args = recv_rpc['kwargs']['fastvideo_args']
                    output_batch = self.execute_forward(forward_batch,
                                                        fastvideo_args)
                    logging_info = None
                    if envs.FASTVIDEO_STAGE_LOGGING:
                        logging_info = output_batch.logging_info
                    self.pipe.send({
                        "output_batch": output_batch.output.cpu(),
                        "logging_info": logging_info
                    })
                elif method_name == 'set_lora_adapter':
                    lora_nickname = recv_rpc['kwargs']['lora_nickname']
                    lora_path = recv_rpc['kwargs']['lora_path']
                    self.set_lora_adapter(lora_nickname, lora_path)
                    logger.info("Worker %d set LoRA adapter %s with path %s",
                                self.rank, lora_nickname, lora_path)
                    self.pipe.send({"status": "lora_adapter_set"})
                elif method_name == 'unmerge_lora_weights':
                    self.unmerge_lora_weights()
                    logger.info("Worker %d unmerged LoRA weights", self.rank)
                    self.pipe.send({"status": "lora_adapter_unmerged"})
                elif method_name == 'merge_lora_weights':
                    self.merge_lora_weights()
                    logger.info("Worker %d merged LoRA weights", self.rank)
                    self.pipe.send({"status": "lora_adapter_merged"})
                else:
                    # Handle other methods dynamically if needed
                    args = recv_rpc.get('args', ())
                    kwargs = recv_rpc.get('kwargs', {})
                    if hasattr(self, method_name):
                        method = getattr(self, method_name)
                        result = method(*args, **kwargs)
                        self.pipe.send(result)
                    else:
                        self.pipe.send(
                            {"error": f"Unknown method: {method_name}"})
            except KeyboardInterrupt:
                logger.error(
                    "Worker %d in loop received KeyboardInterrupt, aborting forward pass",
                    self.rank)
                try:
                    self.pipe.send(
                        {"error": "Operation aborted by KeyboardInterrupt"})
                    logger.info("Worker %d sent error response after interrupt",
                                self.rank)
                except Exception as e:
                    logger.error("Worker %d failed to send error response: %s",
                                 self.rank, str(e))
                continue


def run_worker_process(fastvideo_args: FastVideoArgs, local_rank: int,
                       rank: int, pipe: Connection, master_port: int):
    # Add process-specific prefix to stdout and stderr
    process_name = mp.current_process().name
    pid = os.getpid()
    _add_prefix(sys.stdout, process_name, pid)
    _add_prefix(sys.stderr, process_name, pid)

    # Config the process
    kill_itself_when_parent_died()
    faulthandler.enable()
    parent_process = psutil.Process().parent()

    logger.info("Worker %d initializing...",
                rank,
                local_main_process_only=False)

    try:
        worker = Worker(fastvideo_args, local_rank, rank, pipe, master_port)
        logger.info("Worker %d sending ready", rank)
        pipe.send({
            "status": "ready",
            "local_rank": local_rank,
        })
        worker.event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error("Worker %d hit an exception: %s", rank, traceback)
        parent_process.send_signal(signal.SIGQUIT)


def _add_prefix(file: TextIO, worker_name: str, pid: int) -> None:
    """Prepend each output line with process-specific prefix"""

    prefix = f"{CYAN}({worker_name} pid={pid}){RESET} "
    file_write = file.write

    def write_with_prefix(s: str):
        if not s:
            return
        if file.start_new_line:  # type: ignore[attr-defined]
            file_write(prefix)
        idx = 0
        while (next_idx := s.find('\n', idx)) != -1:
            next_idx += 1
            file_write(s[idx:next_idx])
            if next_idx == len(s):
                file.start_new_line = True  # type: ignore[attr-defined]
                return
            file_write(prefix)
            idx = next_idx
        file_write(s[idx:])
        file.start_new_line = False  # type: ignore[attr-defined]

    file.start_new_line = True  # type: ignore[attr-defined]
    file.write = write_with_prefix  # type: ignore[method-assign]
