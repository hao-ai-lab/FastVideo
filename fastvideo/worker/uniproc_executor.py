# SPDX-License-Identifier: Apache-2.0
"""In-process executor: one worker in the current process, no spawn."""
from __future__ import annotations

import atexit
import logging
import logging.handlers
from collections.abc import Callable
from queue import Queue
from typing import Any

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines import ForwardBatch
from fastvideo.utils import get_distributed_init_method, get_loopback_ip, get_open_port
from fastvideo.worker.executor import Executor
from fastvideo.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


def _make_queue_log_handler(log_queue: Queue) -> logging.Handler:
    return logging.handlers.QueueHandler(log_queue)


class UniprocExecutor(Executor):
    """Run a single Worker in the current process.

    Used when ``num_gpus == 1`` (including the default ``mp`` backend) or when
    ``distributed_executor_backend == "uni"``. Weights load once; no child
    process is spawned.
    """

    def _init_executor(self) -> None:
        if self.fastvideo_args.num_gpus != 1:
            raise ValueError("UniprocExecutor only supports num_gpus=1. "
                             f"Got num_gpus={self.fastvideo_args.num_gpus}.")

        self.world_size = 1
        self.shutting_down = False
        self._log_queue_handler: logging.Handler | None = None

        master_port = get_open_port(self.fastvideo_args.master_port)
        distributed_init_method = get_distributed_init_method(get_loopback_ip(), master_port)
        logger.info("Initializing UniprocExecutor with master port: %s", master_port)

        self.driver_worker = WorkerWrapperBase(fastvideo_args=self.fastvideo_args, rpc_rank=0)
        self.driver_worker.init_worker([{
            "fastvideo_args": self.fastvideo_args,
            "local_rank": 0,
            "rank": 0,
            "distributed_init_method": distributed_init_method,
        }])
        self.driver_worker.init_device()

        if self._log_queue is not None:
            self.set_log_queue(self._log_queue)

        atexit.register(self.shutdown)

    def execute_forward(self, forward_batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        responses: list[ForwardBatch] = self.collective_rpc("execute_forward",
                                                            kwargs={
                                                                "forward_batch": forward_batch,
                                                                "fastvideo_args": fastvideo_args,
                                                            })
        output_batch = responses[0]
        extra = output_batch.extra or {}
        if torch.cuda.is_available():
            extra["peak_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        output_batch.extra = extra
        return output_batch

    def execute_streaming_reset(self, forward_batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> dict[str, Any]:
        responses: list[dict[str, Any]] = self.collective_rpc(
            "execute_streaming_reset",
            kwargs={
                "forward_batch": forward_batch,
                "fastvideo_args": fastvideo_args,
            },
        )
        return responses[0]

    def execute_streaming_step(self, keyboard_action=None, mouse_action=None) -> ForwardBatch:
        responses: list[ForwardBatch] = self.collective_rpc(
            "execute_streaming_step",
            kwargs={
                "keyboard_action": keyboard_action,
                "mouse_action": mouse_action,
            },
        )
        return responses[0]

    async def execute_streaming_step_async(self, keyboard_action=None, mouse_action=None) -> ForwardBatch:
        return self.execute_streaming_step(keyboard_action, mouse_action)

    def execute_streaming_clear(self) -> dict[str, Any]:
        responses: list[dict[str, Any]] = self.collective_rpc("execute_streaming_clear")
        return responses[0]

    def set_lora_adapter(self,
                         lora_nickname: str,
                         lora_path: str | None = None,
                         strength: float = 1.0,
                         accumulate: bool = False) -> None:
        responses = self.collective_rpc("set_lora_adapter",
                                        kwargs={
                                            "lora_nickname": lora_nickname,
                                            "lora_path": lora_path,
                                            "strength": strength,
                                            "accumulate": accumulate,
                                        })
        for i, response in enumerate(responses):
            if response["status"] != "lora_adapter_set":
                raise RuntimeError(f"Worker {i} failed to set LoRA adapter to {lora_path}")

    def unmerge_lora_weights(self) -> None:
        responses = self.collective_rpc("unmerge_lora_weights", kwargs={})
        for i, response in enumerate(responses):
            if response["status"] != "lora_adapter_unmerged":
                raise RuntimeError(f"Worker {i} failed to unmerge LoRA weights")

    def merge_lora_weights(self) -> None:
        responses = self.collective_rpc("merge_lora_weights", kwargs={})
        for i, response in enumerate(responses):
            if response["status"] != "lora_adapter_merged":
                raise RuntimeError(f"Worker {i} failed to merge LoRA weights")

    def set_log_queue(self, log_queue: Queue | None) -> None:
        """Forward in-process logs to the given queue."""
        self._clear_log_queue_handler()
        self._log_queue = log_queue
        if log_queue is None:
            return
        self._log_queue_handler = _make_queue_log_handler(log_queue)
        logging.getLogger("fastvideo").addHandler(self._log_queue_handler)

    def clear_log_queue(self) -> None:
        self._clear_log_queue_handler()
        self._log_queue = None

    def _clear_log_queue_handler(self) -> None:
        handler = getattr(self, "_log_queue_handler", None)
        if handler is not None:
            logging.getLogger("fastvideo").removeHandler(handler)
            self._log_queue_handler = None

    def collective_rpc(self,
                       method: str | Callable,
                       timeout: float | None = None,
                       args: tuple = (),
                       kwargs: dict | None = None) -> list[Any]:
        del timeout
        kwargs = kwargs or {}
        return [self.driver_worker.execute_method(method, *args, **kwargs)]

    def shutdown(self) -> None:
        if getattr(self, "shutting_down", False):
            return
        self.shutting_down = True
        logger.info("Shutting down UniprocExecutor...")
        self._clear_log_queue_handler()
        worker = getattr(self, "driver_worker", None)
        if worker is not None:
            try:
                worker.shutdown()
            except Exception as e:
                logger.error("Error during UniprocExecutor shutdown: %s", e)
            self.driver_worker = None
        logger.info("UniprocExecutor shutdown complete")

    def __del__(self):
        self.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del exc_type, exc_val, exc_tb
        self.shutdown()
