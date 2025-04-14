from fastvideo.v1.executor.abstract import Executor


class TorchRunExecutor(Executor):

    def _init_executor(self) -> None:
        pass

    def collective_rpc(self, method: Union[str, Callable], timeout: Optional[float] = None, args: tuple = (), kwargs: Optional[dict] = None) -> list[Any]:
        pass
