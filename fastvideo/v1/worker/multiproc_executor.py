import signal
import psutil
from typing import List
from multiprocessing import BaseProcess

from fastvideo.v1.worker.executor import Executor
from fastvideo.v1.inference_args import InferenceArgs
from fastvideo.v1.worker import PipelineWorker
from fastvideo.v1.utils import _init_logger

logger = _init_logger(__name__)


class MultiprocExecutor(Executor):

    def __init__(self, inference_args: InferenceArgs):
        super().__init__(inference_args)

        self.workers = []
        for _ in range(self.inference_args.tp_size):
            self.workers.append(PipelineWorker(self.inference_args))

    def _init_executor(self) -> None:
        # The child processes will send SIGUSR1 when unrecoverable
        # errors happen.
        def sigusr1_handler(signum, frame):
            logger.fatal(
                "MulitprocExecutor got fatal signal from worker processes, "
                "shutting down. See stack trace above for root cause issue.")
            # Propagate error up to parent process.
            parent_process = psutil.Process().parent()
            parent_process.send_signal(signal.SIGUSR1)
            self.shutdown()

        signal.signal(signal.SIGUSR1, sigusr1_handler)

        self.world_size = self.inference_args.num_gpus
        tensor_parallel_size = self.parallel_config.tensor_parallel_size
        self.workers: List[BaseProcess] = []
        for rank in range(self.world_size):
            worker = Worker.make_worker_process()
            self.workers.append(worker)

        self.rpc_broadcast_mq.wait_until_ready()
