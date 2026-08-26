# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from collections.abc import Callable
from queue import Queue
from typing import Any, TypeVar, cast

import fastvideo.envs as envs
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines import ForwardBatch
from fastvideo.utils import init_logger

logger = init_logger(__name__)

_R = TypeVar("_R")
EXTERNAL_LAUNCHER_BACKEND = "external_launcher"


def external_launcher_requested(backend: str) -> bool:
    """Return whether ``backend`` opts into launcher-owned SPMD execution."""
    return backend == EXTERNAL_LAUNCHER_BACKEND or (backend == "mp" and envs.FASTVIDEO_EXTERNAL_LAUNCHER)


def reject_external_launcher(backend: str, *, entrypoint: str) -> None:
    """Reject external-launcher mode from an entrypoint without an SPMD control plane."""
    if external_launcher_requested(backend):
        raise ValueError(
            "external_launcher is supported only for synchronized offline generation; "
            f"it cannot be used by {entrypoint}. Launch `fastvideo generate` or the offline VideoGenerator API instead."
        )


class Executor(ABC):

    def __init__(
        self,
        fastvideo_args: FastVideoArgs,
        *,
        log_queue=None,
    ):
        self.fastvideo_args = fastvideo_args
        self._log_queue = log_queue

        self._init_executor()

    @abstractmethod
    def _init_executor(self) -> None:
        raise NotImplementedError

    @staticmethod
    def get_class(
        fastvideo_args: FastVideoArgs,
        *,
        allow_external_launcher: bool = False,
    ) -> type["Executor"]:
        backend = fastvideo_args.distributed_executor_backend
        if external_launcher_requested(backend):
            if not allow_external_launcher:
                reject_external_launcher(backend, entrypoint="this entrypoint")
            from fastvideo.worker.external_launcher_executor import ExternalLauncherExecutor
            return cast(type["Executor"], ExternalLauncherExecutor)
        if backend == "mp":
            from fastvideo.worker.multiproc_executor import MultiprocExecutor
            return cast(type["Executor"], MultiprocExecutor)
        elif backend == "ray":
            from fastvideo.worker.ray_distributed_executor import RayDistributedExecutor
            return cast(type["Executor"], RayDistributedExecutor)
        else:
            raise ValueError(f"Unsupported distributed executor backend: {backend}")

    @property
    def is_output_rank(self) -> bool:
        """Whether this process owns user-facing generated outputs."""
        return True

    @property
    def uses_spmd_execution(self) -> bool:
        """Whether every launcher process must call this executor in lockstep."""
        return False

    def broadcast_from_output_rank(self, value: _R | None) -> _R:
        """Broadcast a small control-plane value from the output rank.

        Controller-owned executors already have one caller, so their default
        implementation simply returns the supplied value.
        """
        return cast(_R, value)

    def execute_forward(
        self,
        forward_batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        outputs: list[dict[str, Any]] = self.collective_rpc("execute_forward",
                                                            kwargs={
                                                                "forward_batch": forward_batch,
                                                                "fastvideo_args": fastvideo_args
                                                            })
        return cast(ForwardBatch, outputs[0]["output_batch"])

    @abstractmethod
    def set_lora_adapter(self,
                         lora_nickname: str,
                         lora_path: str | None = None,
                         strength: float = 1.0,
                         accumulate: bool = False) -> None:
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
            timeout: Maximum time in seconds to wait for execution. `None`
                means wait indefinitely. Implementations that cannot enforce
                a per-call timeout must reject a non-`None` value explicitly.
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
    def set_log_queue(self, log_queue: Queue | None) -> None:
        """Forward worker logs to the given queue. Call before generate_video."""
        self.collective_rpc("set_log_queue", kwargs={"log_queue": log_queue})

    @abstractmethod
    def clear_log_queue(self) -> None:
        """Stop forwarding worker logs to the queue. Call after generate_video."""
        self.collective_rpc("clear_log_queue")

    @abstractmethod
    def shutdown(self) -> None:
        """
        Shutdown the executor.
        """
        raise NotImplementedError
