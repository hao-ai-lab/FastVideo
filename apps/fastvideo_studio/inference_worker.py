# SPDX-License-Identifier: Apache-2.0
"""Keep Studio inference state out of the API process, including on one GPU."""
from fastvideo import VideoGenerator
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.worker.executor import reject_external_launcher
from fastvideo.worker.multiproc_executor import MultiprocExecutor


class StudioVideoGenerator(VideoGenerator):

    @classmethod
    def from_fastvideo_args(cls, fastvideo_args: FastVideoArgs, *, log_queue=None):
        backend = fastvideo_args.distributed_executor_backend
        reject_external_launcher(backend, entrypoint='Studio')
        if backend != 'mp':
            raise ValueError('Studio requires the mp executor for process isolation')
        return cls(fastvideo_args=fastvideo_args,
                   executor_class=MultiprocExecutor,
                   log_stats=False,
                   log_queue=log_queue)
