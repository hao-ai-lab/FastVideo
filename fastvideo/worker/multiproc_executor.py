# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import atexit
import contextlib
from dataclasses import dataclass
from enum import Enum
import faulthandler
import multiprocessing as mp
from multiprocessing.connection import Connection
from multiprocessing.queues import Queue
import os
import queue
import signal
import time
from collections.abc import Callable
from multiprocessing.process import BaseProcess
from typing import Any, cast

import psutil
import torch

from fastvideo.distributed.parallel_state import get_dp_group, get_tp_group
import fastvideo.envs as envs
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.utils import (decorate_logs, get_distributed_init_method,
                             get_exception_traceback, get_loopback_ip,
                             get_mp_context, get_open_port,
                             kill_itself_when_parent_died, force_spawn)
from fastvideo.worker.executor import Executor
from fastvideo.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


class StreamingTaskType(str, Enum):
    """
    Enumeration for different streaming task types.
    
    Inherits from str to allow string comparison for backward compatibility.
    """
    RESET = "reset"
    STEP = "step"
    CLEAR = "clear"
    EXIT = "exit"


@dataclass
class StreamingTask:
    """Task submitted to worker via input queue."""
    task_type: StreamingTaskType
    # For STEP tasks:
    keyboard_action: torch.Tensor | None = None
    mouse_action: torch.Tensor | None = None
    # For RESET tasks:
    batch: ForwardBatch | None = None
    fastvideo_args: FastVideoArgs | None = None


@dataclass
class StreamingResult:
    """Result returned from worker via output queue."""
    task_type: StreamingTaskType
    output_batch: ForwardBatch | None = None
    error: Exception | None = None


class MultiprocExecutor(Executor):

    def _init_executor(self) -> None:
        self.world_size = self.fastvideo_args.num_gpus
        self.shutting_down = False

        set_multiproc_executor_envs()

        # Check if master_port is provided in fastvideo_args
        master_port = get_open_port(self.fastvideo_args.master_port)
        distributed_init_method = get_distributed_init_method(
            get_loopback_ip(), master_port)
        logger.info("Use master port: %s", master_port)

        # Create streaming queues BEFORE spawning workers
        ctx = get_mp_context()
        self._streaming_input_queue: Queue | None = ctx.Queue()
        self._streaming_output_queue: Queue | None = ctx.Queue()
        self._streaming_enabled = False

        unready_workers: list[UnreadyWorkerProcHandle] = []
        success = False
        try:
            for rank in range(self.world_size):
                unready_workers.append(
                    WorkerMultiprocProc.make_worker_process(
                        fastvideo_args=self.fastvideo_args,
                        local_rank=rank,
                        rank=rank,
                        distributed_init_method=distributed_init_method,
                        streaming_input_queue=self._streaming_input_queue,
                        streaming_output_queue=self._streaming_output_queue,
                    ))

            # Workers must be created before wait_for_ready to avoid
            # deadlock, since worker.init_device() does a device sync.
            self.workers = WorkerMultiprocProc.wait_for_ready(unready_workers)

            success = True
        finally:
            if not success:
                # Clean up the worker procs if there was a failure.
                # Close death_writers first to signal workers to exit
                self._ensure_worker_termination(
                    [uw.proc for uw in unready_workers])

        # Register shutdown on exit
        atexit.register(self.shutdown)

    def execute_forward(self, forward_batch: ForwardBatch,
                        fastvideo_args: FastVideoArgs) -> ForwardBatch:
        responses = self.collective_rpc("execute_forward",
                                        kwargs={
                                            "forward_batch": forward_batch,
                                            "fastvideo_args": fastvideo_args
                                        })
        output = responses[0]["output_batch"]

        logging_info = None
        if envs.FASTVIDEO_STAGE_LOGGING:
            logging_info = responses[0]["logging_info"]
        else:
            logging_info = None

        # Get extra dict (contains audio, etc.)
        extra = responses[0].get("extra", {})

        result_batch = ForwardBatch(data_type=forward_batch.data_type,
                                    output=output,
                                    logging_info=logging_info,
                                    extra=extra)

        return result_batch

    def execute_streaming_reset(
            self, forward_batch: ForwardBatch,
            fastvideo_args: FastVideoArgs) -> dict[str, Any]:
        responses = self.collective_rpc("execute_streaming_reset",
                                        kwargs={
                                            "forward_batch": forward_batch,
                                            "fastvideo_args": fastvideo_args,
                                        })
        return responses[0]

    def execute_streaming_step(self, keyboard_action: Any,
                               mouse_action: Any) -> ForwardBatch:
        responses = self.collective_rpc("execute_streaming_step",
                                        kwargs={
                                            "keyboard_action": keyboard_action,
                                            "mouse_action": mouse_action,
                                        })
        return responses[0]

    async def execute_streaming_step_async(self, keyboard_action: Any,
                                           mouse_action: Any) -> ForwardBatch:
        responses = await self.collective_rpc_async("execute_streaming_step",
                                                    kwargs={
                                                        "keyboard_action":
                                                        keyboard_action,
                                                        "mouse_action":
                                                        mouse_action,
                                                    })
        return responses[0]

    def execute_streaming_clear(self) -> dict[str, Any]:
        responses = self.collective_rpc("execute_streaming_clear")
        return responses[0]

    def enable_streaming(self) -> None:
        if self._streaming_enabled:
            return

        self.collective_rpc("start_streaming_queue_loop")
        self._streaming_enabled = True

    def disable_streaming(self) -> None:
        if not self._streaming_enabled:
            return

        if self._streaming_input_queue is not None:
            self._streaming_input_queue.put(
                StreamingTask(task_type=StreamingTaskType.EXIT))
        self._streaming_enabled = False

        self._streaming_input_queue = None
        self._streaming_output_queue = None

    def submit_reset(self, forward_batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> None:
        if not self._streaming_enabled:
            self.enable_streaming()

        self._streaming_input_queue.put(
            StreamingTask(
                task_type=StreamingTaskType.RESET,
                batch=forward_batch,
                fastvideo_args=fastvideo_args,
            ))

    def submit_step(self, keyboard_action: torch.Tensor | None,
                    mouse_action: torch.Tensor | None) -> None:
        if not self._streaming_enabled:
            raise RuntimeError(
                "Streaming mode not enabled. Call enable_streaming() first.")

        self._streaming_input_queue.put(
            StreamingTask(
                task_type=StreamingTaskType.STEP,
                keyboard_action=keyboard_action,
                mouse_action=mouse_action,
            ))

    def submit_clear(self) -> None:
        if self._streaming_enabled and self._streaming_input_queue is not None:
            self._streaming_input_queue.put(
                StreamingTask(task_type=StreamingTaskType.CLEAR))

    def get_result(self,
                   timeout: float | None = None) -> StreamingResult | None:
        if not self._streaming_enabled or self._streaming_output_queue is None:
            return None

        try:
            if timeout == 0:
                return self._streaming_output_queue.get_nowait()
            else:
                return self._streaming_output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def wait_result(self) -> StreamingResult:
        if not self._streaming_enabled or self._streaming_output_queue is None:
            raise RuntimeError("Streaming mode not enabled.")

        return self._streaming_output_queue.get()

    def set_lora_adapter(self,
                         lora_nickname: str,
                         lora_path: str | None = None) -> None:
        responses = self.collective_rpc("set_lora_adapter",
                                        kwargs={
                                            "lora_nickname": lora_nickname,
                                            "lora_path": lora_path
                                        })
        for i, response in enumerate(responses):
            if response["status"] != "lora_adapter_set":
                raise RuntimeError(
                    f"Worker {i} failed to set LoRA adapter to {lora_path}")

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

    def collective_rpc(self,
                       method: str | Callable,
                       timeout: float | None = None,
                       args: tuple = (),
                       kwargs: dict | None = None) -> list[Any]:
        kwargs = kwargs or {}

        try:
            for worker in self.workers:
                worker.pipe.send({
                    "method": method,
                    "args": args,
                    "kwargs": kwargs
                })

            responses = []
            for worker in self.workers:
                response = worker.pipe.recv()
                responses.append(response)
            return responses
        except TimeoutError as e:
            raise TimeoutError(f"RPC call to {method} timed out.") from e
        except KeyboardInterrupt as e:
            # if we catch a KeyboardInterrupt, user wants to stop the execution.
            # we need to send a signal to all workers to stop.
            logger.info(
                "Received KeyboardInterrupt, sending SIGINT to all workers")
            for worker in self.workers:
                if worker.proc.pid is not None:
                    os.kill(worker.proc.pid, signal.SIGINT)
            raise e
        except Exception as e:
            raise e

    async def collective_rpc_async(self,
                                   method: str | Callable,
                                   timeout: float | None = None,
                                   args: tuple = (),
                                   kwargs: dict | None = None) -> list[Any]:
        kwargs = kwargs or {}
        loop = asyncio.get_running_loop()

        for worker in self.workers:
            worker.pipe.send({"method": method, "args": args, "kwargs": kwargs})

        async def recv_from_worker(worker: WorkerProcHandle) -> Any:
            return await loop.run_in_executor(None, worker.pipe.recv)

        responses = await asyncio.gather(
            *[recv_from_worker(worker) for worker in self.workers])
        return list(responses)

    def shutdown(self) -> None:
        """Properly shut down the executor and its workers"""
        if hasattr(self, 'shutting_down') and self.shutting_down:
            return  # Prevent multiple shutdown calls

        logger.info("Shutting down MultiprocExecutor...")
        self.shutting_down = True

        # First try gentle termination
        try:
            # Send termination message to all workers
            for worker in self.workers:
                with contextlib.suppress(Exception):
                    worker.pipe.send({
                        "method": "shutdown",
                        "args": (),
                        "kwargs": {}
                    })

            # Give workers some time to exit gracefully
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < 5.0:  # 5 seconds timeout
                if all(not worker.proc.is_alive() for worker in self.workers):
                    break
                time.sleep(0.1)

            # Force terminate any remaining workers
            for worker in self.workers:
                if worker.proc.is_alive():
                    worker.proc.terminate()

            # Final timeout for terminate
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < 2.0:  # 2 seconds timeout
                if all(not worker.proc.is_alive() for worker in self.workers):
                    break
                time.sleep(0.1)

            # Kill if still alive
            for worker in self.workers:
                if worker.proc.is_alive():
                    worker.proc.kill()
                worker.proc.join(timeout=1.0)

        except Exception as e:
            logger.error("Error during shutdown: %s", e)
            # Last resort, try to kill all workers
            for worker in self.workers:
                with contextlib.suppress(Exception):
                    if worker.proc.is_alive():
                        worker.proc.kill()

        # Clean up pipes
        for worker in self.workers:
            with contextlib.suppress(Exception):
                worker.pipe.close()

        self.workers = []
        logger.info("MultiprocExecutor shutdown complete")

    @staticmethod
    def _ensure_worker_termination(worker_procs: list[BaseProcess]):
        """Ensure that all worker processes are terminated. Assumes workers have
        received termination requests. Waits for processing, then sends
        termination and kill signals if needed."""

        def wait_for_termination(procs: list[BaseProcess],
                                 timeout: float) -> bool:
            if not time:
                # If we are in late stage shutdown, the interpreter may replace
                # `time` with `None`.
                return all(not proc.is_alive() for proc in procs)
            start_time = time.time()
            while time.time() - start_time < timeout:
                if all(not proc.is_alive() for proc in procs):
                    return True
                time.sleep(0.1)
            return False

        # Send SIGTERM if still running
        active_procs = [proc for proc in worker_procs if proc.is_alive()]
        for p in active_procs:
            p.terminate()
        if not wait_for_termination(active_procs, 4):
            # Send SIGKILL if still running
            active_procs = [p for p in active_procs if p.is_alive()]
            for p in active_procs:
                p.kill()

    def __del__(self):
        """Ensure cleanup on garbage collection"""
        self.shutdown()

    def __enter__(self):
        """Support for context manager protocol"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure cleanup when exiting context"""
        self.shutdown()


@dataclass
class UnreadyWorkerProcHandle:
    """WorkerProcess handle before READY."""
    proc: BaseProcess
    rank: int
    pipe: Connection
    ready_pipe: Connection


@dataclass
class WorkerProcHandle:
    proc: BaseProcess
    rank: int
    pipe: Connection

    @classmethod
    def from_unready_handle(
            cls, unready_handle: UnreadyWorkerProcHandle) -> WorkerProcHandle:
        return cls(
            proc=unready_handle.proc,
            rank=unready_handle.rank,
            pipe=unready_handle.pipe,
        )


class WorkerMultiprocProc:
    """Adapter that runs one Worker in busy loop."""

    READY_STR = "READY"

    def __init__(
        self,
        fastvideo_args: FastVideoArgs,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        pipe: Connection,
        streaming_input_queue: Queue | None = None,
        streaming_output_queue: Queue | None = None,
    ):
        self.rank = rank
        self.pipe = pipe
        self.streaming_input_queue = streaming_input_queue
        self.streaming_output_queue = streaming_output_queue
        wrapper = WorkerWrapperBase(fastvideo_args=fastvideo_args,
                                    rpc_rank=rank)

        all_kwargs: list[dict] = [{} for _ in range(fastvideo_args.num_gpus)]
        all_kwargs[rank] = {
            "fastvideo_args": fastvideo_args,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
        }
        wrapper.init_worker(all_kwargs)
        self.worker = wrapper

        # Initialize device
        self.worker.init_device()

        # Set process title and log prefix
        self.setup_proc_title_and_log_prefix()

    @staticmethod
    def make_worker_process(
        fastvideo_args: FastVideoArgs,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        streaming_input_queue: Queue | None = None,
        streaming_output_queue: Queue | None = None,
    ) -> UnreadyWorkerProcHandle:
        context = get_mp_context()
        executor_pipe, worker_pipe = context.Pipe(duplex=True)
        reader, writer = context.Pipe(duplex=False)

        process_kwargs = {
            "fastvideo_args": fastvideo_args,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "pipe": worker_pipe,
            "ready_pipe": writer,
            "streaming_input_queue": streaming_input_queue,
            "streaming_output_queue": streaming_output_queue,
        }
        # Run EngineCore busy loop in background process.
        proc = context.Process(target=WorkerMultiprocProc.worker_main,
                               kwargs=process_kwargs,
                               name=f"FVWorkerProc-{rank}",
                               daemon=True)

        proc.start()
        worker_pipe.close()
        return UnreadyWorkerProcHandle(proc, rank, executor_pipe, reader)

    @staticmethod
    def worker_main(*args, **kwargs):
        """ Worker initialization and execution loops.
        This runs a background process """

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the worker
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        kill_itself_when_parent_died()
        faulthandler.enable()
        parent_process = psutil.Process().parent()

        worker = None
        ready_pipe = kwargs.pop("ready_pipe")
        rank = kwargs.get("rank")

        try:
            worker = WorkerMultiprocProc(*args, **kwargs)

            # Send READY once we know everything is loaded
            ready_pipe.send({
                "status": WorkerMultiprocProc.READY_STR,
            })

            ready_pipe.close()
            ready_pipe = None

            worker.worker_busy_loop()

        except Exception:
            if ready_pipe is not None:
                logger.exception("WorkerMultiprocProc failed to start.")
            else:
                logger.exception("WorkerMultiprocProc failed.")

            # The parent sends a SIGTERM to all worker processes if
            # any worker dies. Set this value so we don't re-throw
            # SystemExit() to avoid zmq exceptions in __del__.
            shutdown_requested = True
            traceback = get_exception_traceback()
            logger.error("Worker %d hit an exception: %s", rank, traceback)
            parent_process.send_signal(signal.SIGQUIT)

        finally:
            if ready_pipe is not None:
                ready_pipe.close()
            # Clean up once worker exits busy loop
            if worker is not None:
                worker.shutdown()

    @staticmethod
    def wait_for_ready(
        unready_proc_handles: list[UnreadyWorkerProcHandle]
    ) -> list[WorkerProcHandle]:

        e = Exception("WorkerMultiprocProc initialization failed due to "
                      "an exception in a background process. "
                      "See stack trace for root cause.")

        pipes = {handle.ready_pipe: handle for handle in unready_proc_handles}
        ready_proc_handles: list[WorkerProcHandle
                                 | None] = ([None] * len(unready_proc_handles))
        while pipes:
            ready = mp.connection.wait(pipes.keys())
            for pipe in ready:
                assert isinstance(pipe, Connection)
                try:
                    # Wait until the WorkerProc is ready.
                    unready_proc_handle = pipes.pop(pipe)
                    response: dict[str, Any] = pipe.recv()
                    if response["status"] != "READY":
                        raise e

                    ready_proc_handles[unready_proc_handle.rank] = (
                        WorkerProcHandle.from_unready_handle(
                            unready_proc_handle))

                except EOFError:
                    e.__suppress_context__ = True
                    raise e from None

                finally:
                    # Close connection.
                    pipe.close()

        logger.info("%d workers ready", len(ready_proc_handles))
        return cast(list[WorkerProcHandle], ready_proc_handles)

    def shutdown(self) -> dict[str, Any]:
        return self.worker.shutdown()

    def worker_busy_loop(self) -> None:
        """Main busy loop for Multiprocessing Workers"""
        while True:
            logger.info("Worker %d starting event loop...", self.rank)
            try:
                rpc_call = self.pipe.recv()
                method = rpc_call.get("method")
                args = rpc_call.get("args", ())
                kwargs = rpc_call.get("kwargs", {})

                if isinstance(method, str):
                    if method == "shutdown":
                        response = self.shutdown()
                        with contextlib.suppress(Exception):
                            self.pipe.send(response)
                        break
                    if method == "start_streaming_queue_loop":
                        self.pipe.send(
                            {"status": "streaming_queue_loop_started"})
                        self.streaming_queue_loop()
                        continue
                    if method == 'execute_forward':
                        forward_batch = kwargs['forward_batch']
                        fastvideo_args = kwargs['fastvideo_args']
                        output_batch = self.worker.execute_forward(
                            forward_batch, fastvideo_args)
                        logging_info = None
                        if envs.FASTVIDEO_STAGE_LOGGING:
                            logging_info = output_batch.logging_info
                        self.pipe.send({
                            "output_batch": output_batch.output.cpu(),
                            "logging_info": logging_info,
                            "extra": output_batch.extra,
                        })
                    else:
                        result = self.worker.execute_method(
                            method, *args, **kwargs)
                        self.pipe.send(result)
                else:
                    result = self.worker.execute_method(method, *args, **kwargs)
                    self.pipe.send(result)
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

    def streaming_queue_loop(self) -> None:
        if self.streaming_input_queue is None or self.streaming_output_queue is None:
            logger.error("Worker %d: streaming queues not initialized",
                         self.rank)
            return

        while True:
            try:
                task: StreamingTask = self.streaming_input_queue.get()

                if task.task_type == StreamingTaskType.EXIT:
                    break
                elif task.task_type == StreamingTaskType.RESET:
                    try:
                        self.worker.execute_streaming_reset(
                            task.batch, task.fastvideo_args)
                        self.streaming_output_queue.put(
                            StreamingResult(task_type=StreamingTaskType.RESET))
                    except Exception as e:
                        logger.error("Worker %d reset error: %s", self.rank, e)
                        self.streaming_output_queue.put(
                            StreamingResult(task_type=StreamingTaskType.RESET,
                                            error=e))
                elif task.task_type == StreamingTaskType.STEP:
                    try:
                        batch = self.worker.execute_streaming_step(
                            task.keyboard_action, task.mouse_action)
                        self.streaming_output_queue.put(
                            StreamingResult(task_type=StreamingTaskType.STEP,
                                            output_batch=batch))
                    except Exception as e:
                        logger.error("Worker %d step error: %s", self.rank, e)
                        self.streaming_output_queue.put(
                            StreamingResult(task_type=StreamingTaskType.STEP,
                                            error=e))
                elif task.task_type == StreamingTaskType.CLEAR:
                    self.worker.execute_streaming_clear()
                    self.streaming_output_queue.put(
                        StreamingResult(task_type=StreamingTaskType.CLEAR))
            except Exception as e:
                logger.error("Worker %d queue loop error: %s", self.rank, e)
                self.streaming_output_queue.put(
                    StreamingResult(task_type=StreamingTaskType.STEP, error=e))

    @staticmethod
    def setup_proc_title_and_log_prefix() -> None:
        dp_size = get_dp_group().world_size
        dp_rank = get_dp_group().rank_in_group
        tp_size = get_tp_group().world_size
        tp_rank = get_tp_group().rank_in_group
        process_name = "Worker"
        if dp_size > 1:
            process_name += f"_DP{dp_rank}"
        if tp_size > 1:
            process_name += f"_TP{tp_rank}"
        decorate_logs(process_name)


def set_multiproc_executor_envs() -> None:
    """ Set up environment variables that should be used when there are workers
    in a multiprocessing environment. This should be called by the parent 
    process before worker processes are created"""

    force_spawn()
