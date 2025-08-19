# SPDX-License-Identifier: Apache-2.0
import atexit
import contextlib
import multiprocessing as mp
import os
import signal
import socket
import time
from collections.abc import Callable
from multiprocessing.process import BaseProcess
from typing import Any

import fastvideo.envs as envs
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.worker.executor import Executor
from fastvideo.worker.gpu_worker import run_worker_process

logger = init_logger(__name__)


class MultiprocExecutor(Executor):

    def _init_executor(self) -> None:
        self.world_size = self.fastvideo_args.num_gpus
        self.shutting_down = False

        # this will force the use of the `spawn` multiprocessing start if cuda
        # is initialized
        mp.set_start_method("spawn", force=True)

        self.workers: list[BaseProcess] = []
        self.worker_pipes = []

        # Check if master_port is provided in fastvideo_args
        if hasattr(
                self.fastvideo_args,
                'master_port') and self.fastvideo_args.master_port is not None:
            self.master_port = self.fastvideo_args.master_port
            logger.info("Using provided master port: %s", self.master_port)
        else:
            # Auto-find available port
            import random
            for port in range(29503 + random.randint(0, 10000), 65535):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    if s.connect_ex(('localhost', port)) != 0:
                        self.master_port = port
                        break
            else:
                raise ValueError("No unused port found to use as master port")
            logger.info("Auto-selected master port: %s", self.master_port)

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
        output = responses[0]["output_batch"]

        logging_info = None
        if envs.FASTVIDEO_STAGE_LOGGING:
            logging_info = responses[0]["logging_info"]
        else:
            logging_info = None

        result_batch = ForwardBatch(data_type=forward_batch.data_type,
                                    output=output,
                                    logging_info=logging_info)

        return result_batch

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
