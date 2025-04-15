from abc import ABC, abstractmethod

import torch
from fastvideo.v1.inference_args import InferenceArgs
from fastvideo.v1.pipelines import ForwardBatch


class Executor(ABC):

    def __init__(self, inference_args: InferenceArgs):
        self.inference_args = inference_args

    @abstractmethod
    def _init_executor(self) -> None:
        raise NotImplementedError

    @classmethod
    def get_class(cls, inference_args: InferenceArgs) -> type["Executor"]:
        if inference_args.distributed_executor_backend == "torch":
            from fastvideo.v1.worker.torchrun_executor import TorchRunExecutor
            return TorchRunExecutor(inference_args)
        elif inference_args.distributed_executor_backend == "mp":
            from fastvideo.v1.worker.multiproc_executor import MultiprocExecutor
            return MultiprocExecutor(inference_args)
        else:
            raise ValueError(
                f"Unsupported distributed executor backend: {inference_args.distributed_executor_backend}"
            )

    def execute_forward(self, forward_batch: ForwardBatch) -> torch.Tensor:
        pass
