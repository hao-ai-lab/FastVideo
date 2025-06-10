# SPDX-License-Identifier: Apache-2.0
import atexit
import contextlib
import multiprocessing as mp
import os
import signal
import socket
import time
from multiprocessing.process import BaseProcess
from typing import Any, Callable, List, Optional, Union, cast

from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.worker.executor import Executor
from fastvideo.v1.worker.gpu_worker import run_worker_process

logger = init_logger(__name__)


class MultiprocExecutor(Executor):

    def _init_executor(self) -> None:
        self.world_size = self.fastvideo_args.num_gpus
        self.shutting_down = False

        # this will force the use of the `spawn` multiprocessing start if cuda
        # is initialized
        mp.set_start_method("spawn", force=True)

        self.workers: List[BaseProcess] = []
        self.worker_pipes = []
        self.master_port = None

        for port in range(29503, 65535):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', port)) != 0:
                    self.master_port = port
                    break
        if self.master_port is None:
            raise ValueError("No unused port found to use as master port")

        # Create pipes and start workers
        for rank in range(self.world_size):
            executor_pipe, worker_pipe = mp.Pipe(duplex=True)
            self.worker_pipes.append(executor_pipe)

            worker = mp.Process(target=run_worker_process,
                                name=f"FVWorkerProc-{rank}",
                                kwargs=dict(fastvideo_args=self.fastvideo_args,
                                            local_rank=rank,
                                            rank=rank,
                                            pipe=worker_pipe,
                                            master_port=self.master_port))
            worker.start()
            self.workers.append(worker)

        # Wait for all workers to be ready
        for idx, pipe in enumerate(self.worker_pipes):
            data = pipe.recv()
            if data["status"] != "ready" or data["local_rank"] != idx:
                raise RuntimeError(f"Worker {idx} failed to start")
        logger.info("%d workers ready", self.world_size)

        # Register shutdown on exit
        atexit.register(self.shutdown)

    def execute_forward(self, forward_batch: ForwardBatch,
                        fastvideo_args: FastVideoArgs) -> ForwardBatch:
        responses = self.collective_rpc("execute_forward",
                                        kwargs={
                                            "forward_batch": forward_batch,
                                            "fastvideo_args": fastvideo_args
                                        })
        return cast(ForwardBatch, responses[0]["output_batch"])

    def set_lora_adapter(self, lora_nickname: str, lora_path: str) -> None:
        self.collective_rpc("set_lora_adapter",
                            kwargs={
                                "lora_nickname": lora_nickname,
                                "lora_path": lora_path
                            })

    def collective_rpc(self,
                       method: Union[str, Callable],
                       timeout: Optional[float] = None,
                       args: tuple = (),
                       kwargs: Optional[dict] = None) -> list[Any]:
        kwargs = kwargs or {}

        try:
            for pipe in self.worker_pipes:
                pipe.send({"method": method, "args": args, "kwargs": kwargs})

            responses = []
            for pipe in self.worker_pipes:
                response = pipe.recv()
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
                if worker.pid is not None:
                    os.kill(worker.pid, signal.SIGINT)
            raise e
        except Exception as e:
            raise e

    def shutdown(self) -> None:
        """Properly shut down the executor and its workers"""
        if hasattr(self, 'shutting_down') and self.shutting_down:
            return  # Prevent multiple shutdown calls

        logger.info("Shutting down MultiprocExecutor...")
        self.shutting_down = True

        # First try gentle termination
        try:
            # Send termination message to all workers
            for pipe in self.worker_pipes:
                with contextlib.suppress(Exception):
                    pipe.send({"method": "shutdown", "args": (), "kwargs": {}})

            # Give workers some time to exit gracefully
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < 5.0:  # 5 seconds timeout
                if all(not worker.is_alive() for worker in self.workers):
                    break
                time.sleep(0.1)

            # Force terminate any remaining workers
            for worker in self.workers:
                if worker.is_alive():
                    worker.terminate()

            # Final timeout for terminate
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < 2.0:  # 2 seconds timeout
                if all(not worker.is_alive() for worker in self.workers):
                    break
                time.sleep(0.1)

            # Kill if still alive
            for worker in self.workers:
                if worker.is_alive():
                    worker.kill()
                worker.join(timeout=1.0)

        except Exception as e:
            logger.error("Error during shutdown: %s", e)
            # Last resort, try to kill all workers
            for worker in self.workers:
                with contextlib.suppress(Exception):
                    if worker.is_alive():
                        worker.kill()

        # Clean up pipes
        for pipe in self.worker_pipes:
            with contextlib.suppress(Exception):
                pipe.close()

        self.workers = []
        self.worker_pipes = []
        logger.info("MultiprocExecutor shutdown complete")

    def __del__(self):
        """Ensure cleanup on garbage collection"""
        self.shutdown()

    def __enter__(self):
        """Support for context manager protocol"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure cleanup when exiting context"""
        self.shutdown()
