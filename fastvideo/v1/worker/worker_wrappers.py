class WorkerWrapper:

    def __init__(self, worker: Worker):
        self.worker = worker

    def __call__(self, *args, **kwargs):
        return self.worker(*args, **kwargs)
