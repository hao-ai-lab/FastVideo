# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Union
import os

from fastvideo.v1.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.fastvideo_args import FastVideoArgs, set_current_fastvideo_args
from fastvideo.v1.utils import (update_environment_variables,
                                resolve_obj_by_qualname, run_method)
from fastvideo.v1.logger import init_logger

logger = init_logger(__name__)


class WorkerBase:

    def __init__(self,
                 fastvideo_args: FastVideoArgs,
                 local_rank: int,
                 rank: int,
                 distributed_init_method: str,
                 is_driver_worker: bool = False):
        self.fastvideo_args = fastvideo_args
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

    def init_device(self) -> None:
        raise NotImplementedError

    def get_pipeline(self) -> ComposedPipelineBase:
        raise NotImplementedError

    def execute_pipeline(self, forward_batch: ForwardBatch) -> ForwardBatch:
        raise NotImplementedError

    # TODO: add lora support
    # def add_lora(self, lora_request: LoRARequest) -> bool:
    #     raise NotImplementedError

    # def remove_lora(self, lora_id: int) -> bool:
    #     raise NotImplementedError

    # def pin_lora(self, lora_id: int) -> bool:
    #     raise NotImplementedError

    # def list_loras(self) -> Set[int]:
    #     raise NotImplementedError


class WorkerWrapperBase:
    """
    This class represents one process in an executor/engine. It is responsible
    for lazily initializing the worker and handling the worker's lifecycle.
    We first instantiate the WorkerWrapper, which remembers the worker module
    and class name. Then, when we call `update_environment_variables`, and the
    real initialization happens in `init_worker`.
    """

    def __init__(
        self,
        fastvideo_args: FastVideoArgs,
        rpc_rank: int = 0,
    ) -> None:
        """
        Initialize the worker wrapper with the given fastvideo_args and
        rpc_rank.  Note: rpc_rank is the rank of the worker in the executor. In
        most cases, it is also the rank of the worker in the distributed group.
        However, when multiple executors work together, they can be different.
        e.g. in the case of SPMD-style offline inference with TP=2, users can
        launch 2 engines/executors, each with only 1 worker.  All workers have
        rpc_rank=0, but they have different ranks in the TP group.
        """
        self.rpc_rank = rpc_rank
        self.worker: Optional[WorkerBase] = None

    def adjust_rank(self, rank_mapping: Dict[int, int]) -> None:
        """
        Adjust the rpc_rank based on the given mapping.
        It is only used during the initialization of the executor,
        to adjust the rpc_rank of workers after we create all workers.
        """
        if self.rpc_rank in rank_mapping:
            self.rpc_rank = rank_mapping[self.rpc_rank]

    def update_environment_variables(self, envs_list: List[Dict[str,
                                                                str]]) -> None:
        envs = envs_list[self.rpc_rank]
        key = 'CUDA_VISIBLE_DEVICES'
        if key in envs and key in os.environ:
            # overwriting CUDA_VISIBLE_DEVICES is desired behavior
            # suppress the warning in `update_environment_variables`
            del os.environ[key]
        update_environment_variables(envs)

    def init_worker(self, all_kwargs: List[Dict[str, Any]]) -> None:
        """
        Here we inject some common logic before initializing the worker.
        Arguments are passed to the worker class constructor.
        """
        kwargs = all_kwargs[self.rpc_rank]
        self.fastvideo_args = kwargs.get("fastvideo_args", None)
        assert self.fastvideo_args is not None, (
            "fastvideo_args is required to initialize the worker")
        # enable_trace_function_call_for_thread(self.fastvideo_args)

        if isinstance(self.fastvideo_args.worker_cls, str):
            worker_class = resolve_obj_by_qualname(
                self.fastvideo_args.worker_cls)
        else:
            raise ValueError(
                "worker_cls must be a string of a qualified module name")

        with set_current_fastvideo_args(self.fastvideo_args):
            # To make fastvideo config available during worker initialization
            self.worker = worker_class(**kwargs)
            assert self.worker is not None

    def init_device(self):
        with set_current_fastvideo_args(self.fastvideo_args):
            # To make fastvideo config available during device initialization
            self.worker.init_device()  # type: ignore

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        try:
            # method resolution order:
            # if a method is defined in this class, it will be called directly.
            # otherwise, since we define `__getattr__` and redirect attribute
            # query to `self.worker`, the method will be called on the worker.
            return run_method(self, method, args, kwargs)
        except Exception as e:
            # if the driver worker also execute methods,
            # exceptions in the rest worker may cause deadlock in rpc like ray
            # see https://github.com/vllm-project/vllm/issues/3455
            # print the error and inform the user to solve the error
            msg = (f"Error executing method {method!r}. "
                   "This might cause deadlock in distributed execution.")
            logger.exception(msg)
            raise e

    def __getattr__(self, attr):
        return getattr(self.worker, attr)
