import signal
import psutil
import multiprocessing as mp
import time
import pickle
import cloudpickle
import zmq
from typing import List, Callable, Any, Optional, Union
from multiprocessing.process import BaseProcess

from fastvideo.v1.worker.executor import Executor
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.worker.gpu_worker import run_worker_process
from fastvideo.v1.utils import get_zmq_socket

logger = init_logger(__name__)


class MultiprocExecutor(Executor):

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

        self.world_size = self.fastvideo_args.num_gpus

        # Set mp start method
        mp.set_start_method("spawn", force=True)

        self.workers: List[BaseProcess] = []
        self.worker_pipe_readers: List[mp.Pipe] = []
        for rank in range(self.world_size):
            reader, writer = mp.Pipe(duplex=False)
            self.worker_pipe_readers.append(reader)

            worker = mp.Process(target=run_worker_process,
                                     args=(self.fastvideo_args, rank, rank,
                                            writer))
            worker.start()
            self.workers.append(worker)
        logger.info(f"Workers: {self.workers}")
        for reader in self.worker_pipe_readers:
            data = reader.recv()
            assert data["status"] == "ready"
            logger.info(f"Worker {data['local_rank']} ready")

        context = zmq.Context(2)
        self.send_to_rpc = get_zmq_socket(
            context, zmq.DEALER, "ipc://fastvideo_rpc_broadcast", True
        )
        logger.info("Sending wait_until_ready to workers")
        self.send_to_rpc.send_pyobj("wait_until_ready")
        self.send_to_rpc.send_pyobj("wait_until_ready")
        logger.info("Sent wait_until_ready to workers")
        recv_req = self.send_to_rpc.recv_pyobj(zmq.BLOCKY)
        print(f"Received wait_until_ready from workers: {recv_req}")
        recv_req = self.send_to_rpc.recv_pyobj(zmq.BLOCKY)
        print(f"Received wait_until_ready from workers: {recv_req}")




        # self.collective_rpc("wait_until_ready")

    def collective_rpc(self,
                       method: Union[str, Callable],
                       timeout: Optional[float] = None,
                       args: tuple = (),
                       kwargs: Optional[dict] = None) -> list[Any]:
        start_time = time.monotonic()
        kwargs = kwargs or {}

        # NOTE: If the args are heterogeneous, then we pack them into a list,
        # and unpack them in the method of every worker, because every worker
        # knows their own rank.
        try:
            if isinstance(method, str):
                send_method = method
            else:
                send_method = cloudpickle.dumps(
                    method, protocol=pickle.HIGHEST_PROTOCOL)
            self.rpc_broadcast_mq.enqueue((send_method, args, kwargs))

            responses = [None] * self.world_size
            for w in self.workers:
                dequeue_timeout = timeout - (time.monotonic() - start_time
                                             ) if timeout is not None else None
                status, result = w.worker_response_mq.dequeue(
                    timeout=dequeue_timeout)

                if status != WorkerProc.ResponseStatus.SUCCESS:
                    if isinstance(result, Exception):
                        raise result
                    else:
                        raise RuntimeError("Worker failed")

                responses[w.rank] = result

            return responses
        except TimeoutError as e:
            raise TimeoutError(f"RPC call to {method} timed out.") from e
        except Exception as e:
            # Re-raise any other exceptions
            raise e