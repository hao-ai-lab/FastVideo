from abc import ABC, abstractmethod

import torch
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.pipelines import ForwardBatch


class Executor(ABC):

    def __init__(self, fastvideo_args: FastVideoArgs):
        self.fastvideo_args = fastvideo_args

    @abstractmethod
    def _init_executor(self) -> None:
        raise NotImplementedError

    @classmethod
    def get_class(cls, fastvideo_args: FastVideoArgs) -> type["Executor"]:
        if fastvideo_args.distributed_executor_backend == "torch":
            from fastvideo.v1.worker.torchrun_executor import TorchRunExecutor
            return TorchRunExecutor(fastvideo_args)
        elif fastvideo_args.distributed_executor_backend == "mp":
            from fastvideo.v1.worker.multiproc_executor import MultiprocExecutor
            return MultiprocExecutor(fastvideo_args)
        else:
            raise ValueError(
                f"Unsupported distributed executor backend: {fastvideo_args.distributed_executor_backend}"
            )

    def execute_forward(self, forward_batch: ForwardBatch) -> torch.Tensor:
        pass
