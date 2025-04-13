# SPDX-License-Identifier: Apache-2.0

from typing import Union

from fastvideo.v1.fastvideo_args import FastVideoArgs

import time
from abc import ABC, abstractmethod
from typing import (Any, Callable, Dict, List, Optional, Tuple)

import torch.nn as nn
from typing_extensions import TypeVar

from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
# from vllm.lora.request import LoRARequest
# from vllm.model_executor.layers.sampler import SamplerOutput
# from vllm.prompt_adapter.request import PromptAdapterRequest
# from vllm.sequence import ExecuteModelRequest, PoolerOutput
# from vllm.utils import make_async
from fastvideo.v1.worker.worker_base import WorkerBase

logger = init_logger(__name__)

_R = TypeVar("_R", default=Any)


class Executor(ABC):
    """Base class for all executors.

    An executor is responsible for executing the model on one device,
    or it can be a distributed executor 
    that can execute the model on multiple devices.
    """

    uses_ray: bool  # whether the executor uses Ray for orchestration.

    def __init__(
        self,
        fastvideo_args: FastVideoArgs,
    ) -> None:
        self.fastvideo_args = fastvideo_args
        self._init_executor()
        self.is_sleeping = False

    @abstractmethod
    def _init_executor(self) -> None:
        raise NotImplementedError

    @staticmethod
    def get_class(fastvideo_args: FastVideoArgs) -> type["Executor"]:
        executor_class: type[Executor]
        distributed_executor_backend = (
            fastvideo_args.distributed_executor_backend)
        # distributed_executor_backend must be set in VllmConfig.__post_init__
        if isinstance(distributed_executor_backend, type):
            if not issubclass(distributed_executor_backend, Executor):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"Executor. Got {distributed_executor_backend}.")
            executor_class = distributed_executor_backend
        elif distributed_executor_backend == "ray":
            from fastvideo.v1.executor.ray_distributed_executor import (  # noqa
                RayDistributedExecutor)
            executor_class = RayDistributedExecutor
        elif distributed_executor_backend == "mp":
            from fastvideo.v1.executor.multiproc_executor import MultiprocExecutor
            executor_class = MultiprocExecutor
        elif distributed_executor_backend == "torch":
            raise NotImplementedError
            executor_class = TorchExecutor
        elif distributed_executor_backend == "external_launcher":
            raise NotImplementedError
            executor_class = ExecutorWithExternalLauncher
        else:
            raise ValueError("Unknown distributed executor backend: "
                             f"{distributed_executor_backend}")
        return executor_class

    @abstractmethod
    def collective_rpc(self,
                       method: Union[str, Callable[..., _R]],
                       timeout: Optional[float] = None,
                       args: Tuple = (),
                       kwargs: Optional[Dict[str, Any]] = None) -> List[_R]:
        """
        Execute an RPC call on all workers.

        Args:
            method: Name of the worker method to execute, or a callable that
                is serialized and sent to all workers to execute.

                If the method is a callable, it should accept an additional
                `self` argument, in addition to the arguments passed in `args`
                and `kwargs`. The `self` argument will be the worker object.
            timeout: Maximum time in seconds to wait for execution. Raises a
                :exc:`TimeoutError` on timeout. `None` means wait indefinitely.
            args: Positional arguments to pass to the worker method.
            kwargs: Keyword arguments to pass to the worker method.

        Returns:
            A list containing the results from each worker.
        
        Note:
            It is recommended to use this API to only pass control messages,
            and set up data-plane communication to pass data.
        """
        raise NotImplementedError

    def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]:
        """
        Run a function directly on the model inside each worker,
        returning the result for each of them.
        """

        def rpc_func(worker: WorkerBase) -> _R:
            return func(worker.get_model())

        return self.collective_rpc(rpc_func)

    def execute_model(self,
                      forward_batch: ForwardBatch) -> Optional[ForwardBatch]:
        output = self.collective_rpc("execute_model", args=(forward_batch, ))
        return output[0]

    def stop_remote_worker_execution_loop(self) -> None:
        """Releases parallel workers from model loop."""
        return

    # TODO: add lora support
    # def add_lora(self, lora_request: LoRARequest) -> bool:
    #     assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
    #     return all(self.collective_rpc("add_lora", args=(lora_request, )))

    # def remove_lora(self, lora_id: int) -> bool:
    #     assert lora_id > 0, "lora_id must be greater than 0."
    #     return all(self.collective_rpc("remove_lora", args=(lora_id, )))

    # def pin_lora(self, lora_id: int) -> bool:
    #     assert lora_id > 0, "lora_id must be greater than 0."
    #     return all(self.collective_rpc("pin_lora", args=(lora_id, )))

    # def list_loras(self) -> Set[int]:
    #     sets = self.collective_rpc("list_loras")
    #     for s in sets:
    #         assert s == sets[0], "All workers should have the same LORAs."
    #     return sets[0]

    def profile(self, is_start: bool = True):
        self.collective_rpc("profile", args=(is_start, ))

    # def start_profile(self) -> None:
    #     self.collective_rpc("start_profile")

    # def stop_profile(self) -> None:
    #     self.collective_rpc("stop_profile")

    def sleep(self, level: int = 1):
        if self.is_sleeping:
            logger.warning("Executor is already sleeping.")
            return
        time_before_sleep = time.perf_counter()
        self.collective_rpc("sleep", kwargs=dict(level=level))
        time_after_sleep = time.perf_counter()
        self.is_sleeping = True
        logger.info("It took %.6f seconds to fall asleep.",
                    time_after_sleep - time_before_sleep)

    def wake_up(self):
        if not self.is_sleeping:
            logger.warning("Executor is not sleeping.")
            return
        time_before_wakeup = time.perf_counter()
        self.collective_rpc("wake_up")
        time_after_wakeup = time.perf_counter()
        self.is_sleeping = False
        logger.info("It took %.6f seconds to wake up.",
                    time_after_wakeup - time_before_wakeup)

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        self.collective_rpc("save_sharded_state",
                            kwargs=dict(path=path,
                                        pattern=pattern,
                                        max_size=max_size))

    @abstractmethod
    def check_health(self) -> None:
        """Checks if the executor is healthy. If not, it should raise an
        exception."""
        raise NotImplementedError

    def shutdown(self) -> None:
        """Shutdown the executor."""
        return

    def __del__(self):
        self.shutdown()

    # async def execute_model_async(
    #         self,
    #         execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
    #     """Executes one model step on the given sequences."""
    #     output = await make_async(self.execute_model)(execute_model_req)
    #     return output

    async def stop_remote_worker_execution_loop_async(self) -> None:
        """Releases parallel workers from model loop."""
        return

    async def check_health_async(self) -> None:
        """Checks if the executor is healthy. If not, it should raise an
        exception."""
        self.check_health()


# class Executor(ExecutorBase):
#     """
#     Abstract class for v1 executors, mainly define some methods for v1.
#     For methods shared by v0 and v1, define them in ExecutorBase"""

#     def initialize_from_config(self,
#                                kv_cache_configs: list[KVCacheConfig]) -> None:
#         """
#         Initialize the KV caches and begin the model execution loop of the
#         underlying workers.
#         """
#         self.collective_rpc("initialize_from_config", args=(kv_cache_configs, ))
#         self.collective_rpc("compile_or_warm_up_model")

#     def determine_available_memory(self) -> list[int]:  # in bytes
#         output = self.collective_rpc("determine_available_memory")
#         return output

#     def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
#         output = self.collective_rpc("get_kv_cache_spec")
#         return output

#     def execute_model(
#         self,
#         scheduler_output,
#     ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
#         output = self.collective_rpc("execute_model", args=(scheduler_output, ))
#         return output[0]

#     @property
#     def max_concurrent_batches(self) -> int:
#         return 1

#     def profile(self, is_start: bool = True):
#         self.collective_rpc("profile", args=(is_start, ))
