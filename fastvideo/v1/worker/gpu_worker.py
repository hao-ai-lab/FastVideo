import contextlib
import faulthandler
import gc
import multiprocessing as mp
import os
import signal
import sys
from typing import Any, Dict, Optional, TextIO, cast

import psutil
import torch

from fastvideo.v1.distributed import (cleanup_dist_env_and_memory,
                                      init_distributed_environment,
                                      initialize_model_parallel)
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines import ForwardBatch, build_pipeline
from fastvideo.v1.utils import (get_exception_traceback,
                                kill_itself_when_parent_died)

logger = init_logger(__name__)

# ANSI color codes
CYAN = '\033[1;36m'
RESET = '\033[0;0m'


class Worker:

    def __init__(self, fastvideo_args: FastVideoArgs, local_rank: int,
                 rank: int, pipe):
        self.fastvideo_args = fastvideo_args
        self.local_rank = local_rank
        self.rank = rank
        # TODO(will): don't hardcode this
        self.distributed_init_method = "env://"
        self.pipe = pipe

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
        assert self.fastvideo_args.device_str is not None
        if self.fastvideo_args.device_str.startswith("cuda"):
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)

            # _check_if_gpu_supports_dtype(self.model_config.dtype)
            gc.collect()
            torch.cuda.empty_cache()
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            raise ValueError(
                f"Unsupported device: {self.fastvideo_args.device_str}")

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29503"
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["RANK"] = str(self.rank)

        # Initialize the distributed environment.
        init_worker_distributed_environment(self.fastvideo_args, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)

        self.pipeline = build_pipeline(self.fastvideo_args)

    def execute_forward(self, forward_batch: ForwardBatch,
                        fastvideo_args: FastVideoArgs) -> ForwardBatch:
        output_batch = self.pipeline.forward(forward_batch, self.fastvideo_args)
        return cast(ForwardBatch, output_batch)

    def shutdown(self) -> Dict[str, Any]:
        """Gracefully shut down the worker process"""
        logger.info("Worker %d shutting down...",
                    self.rank,
                    local_main_process_only=False)
        # Clean up resources
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            # Clean up pipeline resources if needed
            pass
        # Release CUDA resources
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Destroy the distributed environment
        cleanup_dist_env_and_memory(shutdown_ray=False)

        logger.info("Worker %d shutdown complete",
                    self.rank,
                    local_main_process_only=False)
        return {"status": "shutdown_complete"}

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
                    gc.collect()
                    torch.cuda.empty_cache()
                    forward_batch = recv_rpc['kwargs']['forward_batch']
                    fastvideo_args = recv_rpc['kwargs']['fastvideo_args']
                    output_batch = self.execute_forward(forward_batch,
                                                        fastvideo_args)
                    self.pipe.send({"output_batch": output_batch.output.cpu()})
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
                continue


def init_worker_distributed_environment(
    fastvideo_args: FastVideoArgs,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize distributed environment and model parallelism."""

    world_size = fastvideo_args.num_gpus

    torch.cuda.set_device(local_rank)
    init_distributed_environment(world_size=world_size,
                                 rank=rank,
                                 local_rank=local_rank)
    device_str = f"cuda:{local_rank}"
    fastvideo_args.device_str = device_str
    fastvideo_args.device = torch.device(device_str)
    assert fastvideo_args.sp_size is not None
    assert fastvideo_args.tp_size is not None
    initialize_model_parallel(
        sequence_model_parallel_size=fastvideo_args.sp_size,
        tensor_model_parallel_size=fastvideo_args.tp_size,
    )


def run_worker_process(fastvideo_args: FastVideoArgs, local_rank: int,
                       rank: int, pipe):
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
        worker = Worker(fastvideo_args, local_rank, rank, pipe)
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
