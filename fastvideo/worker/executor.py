# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar, cast

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines import ForwardBatch
from fastvideo.utils import init_logger

logger = init_logger(__name__)

_R = TypeVar("_R")


class Executor(ABC):

    def __init__(self, fastvideo_args: FastVideoArgs):
        self.fastvideo_args = fastvideo_args

        self._init_executor()

    @abstractmethod
    def _init_executor(self) -> None:
        raise NotImplementedError

    @staticmethod
    def get_class(fastvideo_args: FastVideoArgs) -> type["Executor"]:
        if fastvideo_args.distributed_executor_backend == "mp":
            from fastvideo.worker.multiproc_executor import MultiprocExecutor
            return cast(type["Executor"], MultiprocExecutor)
        elif fastvideo_args.distributed_executor_backend == "ray":
            from fastvideo.worker.ray_distributed_executor import RayDistributedExecutor
            return cast(type["Executor"], RayDistributedExecutor)
        else:
            raise ValueError(
                f"Unsupported distributed executor backend: {fastvideo_args.distributed_executor_backend}"
            )

    def execute_forward(
        self,
        forward_batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        outputs: list[dict[str,
                           Any]] = self.collective_rpc("execute_forward",
                                                       kwargs={
                                                           "forward_batch":
                                                           forward_batch,
                                                           "fastvideo_args":
                                                           fastvideo_args
                                                       })
        return cast(ForwardBatch, outputs[0]["output_batch"])

    @abstractmethod
    def set_lora_adapter(self,
                         lora_nickname: str,
                         lora_path: str | None = None) -> None:
        """
        Set the LoRA adapter for the workers.
        """
        raise NotImplementedError

    @abstractmethod
    def unmerge_lora_weights(self) -> None:
        """
        Unmerge the LoRA weights for the workers.
        """
        raise NotImplementedError

    @abstractmethod
    def merge_lora_weights(self) -> None:
        """
        Merge the LoRA weights for the workers.
        """
        raise NotImplementedError

    @abstractmethod
    def collective_rpc(self,
                       method: str | Callable[..., _R],
                       timeout: float | None = None,
                       args: tuple = (),
                       kwargs: dict[str, Any] | None = None) -> list[_R]:
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

    @abstractmethod
    def shutdown(self) -> None:
        """
        Shutdown the executor.
        """
        raise NotImplementedError
