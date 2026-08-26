# SPDX-License-Identifier: Apache-2.0
"""Real two-process Gloo lifecycle smoke for ExternalLauncherExecutor."""

from __future__ import annotations

import argparse
import os

import torch

import fastvideo.worker.external_launcher_executor as external_launcher
from fastvideo.api.schema import GenerationRequest
from fastvideo.entrypoints.video_generator import VideoGenerator
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines import ForwardBatch


class _GlooWorld:

    @property
    def cpu_group(self):
        return torch.distributed.group.WORLD

    def broadcast_object(self, value, src=0):
        values = [value]
        torch.distributed.broadcast_object_list(values, src=src, group=self.cpu_group)
        return values[0]


class _SmokeWorkerWrapper:

    def __init__(self, fastvideo_args, rpc_rank):
        self.fastvideo_args = fastvideo_args
        self.rpc_rank = rpc_rank
        self.rank = rpc_rank
        self.device = torch.device("cpu")
        self.worker = self
        self._shutdown = False

    def init_worker(self, all_kwargs):
        kwargs = all_kwargs[self.rpc_rank]
        self.rank = kwargs["rank"]
        self.local_rank = kwargs["local_rank"]

    def init_device(self):
        torch.distributed.init_process_group(
            backend="gloo",
            init_method="env://",
            rank=self.rank,
            world_size=int(os.environ["WORLD_SIZE"]),
        )

    def execute_method(self, method, *args, **kwargs):
        if method == "rank_failure" and self.rank == 1:
            raise ValueError("injected rank-one control failure")
        if method == "identity":
            return {
                "rank": self.rank,
                "local_rank": self.local_rank,
            }
        raise ValueError(f"unsupported smoke method: {method}")

    def execute_forward(self, forward_batch, fastvideo_args):
        del fastvideo_args
        return forward_batch

    def shutdown(self):
        if self._shutdown:
            return
        self._shutdown = True
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _run(mode: str) -> None:
    external_launcher.WorkerWrapperBase = _SmokeWorkerWrapper
    external_launcher.get_world_group = lambda: _GlooWorld()
    external_launcher.torch.cuda.is_available = lambda: False

    world_size = int(os.environ["WORLD_SIZE"])
    args = FastVideoArgs(model_path="external-launcher-smoke", num_gpus=world_size, sp_size=world_size)
    executor = external_launcher.ExternalLauncherExecutor(args)
    try:
        generator = VideoGenerator.__new__(VideoGenerator)
        generator.executor = executor

        if mode == "rank-exit":
            if executor.rank == 1:
                raise RuntimeError("injected pre-request rank exit")
            torch.distributed.barrier(group=external_launcher.get_world_group().cpu_group)
            raise AssertionError("rank-zero barrier unexpectedly survived the peer exit")

        divergent_request = GenerationRequest(prompt=f"rank-{executor.rank}-request")
        synchronized_request = generator._synchronize_request_prompt(divergent_request)
        assert synchronized_request.prompt == "rank-0-request"

        if mode == "rank-failure":
            def fail_output_setup():
                assert executor.is_output_rank
                raise PermissionError("injected output-rank setup failure")

            try:
                generator._run_on_output_rank(fail_output_setup, description="smoke setup")
            except RuntimeError as error:
                assert "PermissionError: injected output-rank setup failure" in str(error)
            else:
                raise AssertionError("output-rank setup failure was not propagated to every process")

            try:
                executor.collective_rpc("rank_failure")
            except RuntimeError as error:
                assert "rank 1 ValueError: injected rank-one control failure" in str(error)
            else:
                raise AssertionError("rank-local failure was not propagated to every process")
            print(f"coordinated_failure rank={executor.rank}", flush=True)
            return

        identities = executor.collective_rpc("identity")
        assert [identity["rank"] for identity in identities] == list(range(world_size))

        output = ForwardBatch(
            data_type="video",
            output=torch.ones((1, 3, 1, 2, 2)),
            latents=torch.ones(2),
            audio_latents=torch.ones(2),
            extra={
                "audio": torch.ones(8),
                "audio_sample_rate": 24000,
                "ltx2_audio_latents": torch.ones(2),
            },
            trajectory_latents=torch.ones(1),
            trajectory_timesteps=[torch.ones(1)],
            trajectory_decoded=[torch.ones(1)],
        )
        result = executor.execute_forward(output, args)
        if executor.is_output_rank:
            assert result.output is not None and result.output.numel() > 0
            assert result.extra["audio_sample_rate"] == 24000
            assert result.trajectory_latents is not None
        else:
            assert result.output is not None and result.output.numel() == 0
            assert "audio" not in result.extra
            assert "audio_sample_rate" not in result.extra
            assert "ltx2_audio_latents" not in result.extra
            assert result.latents is None
            assert result.audio_latents is None
            assert result.trajectory_latents is None
            assert result.trajectory_timesteps is None
            assert result.trajectory_decoded is None
        print(f"lifecycle_ok rank={executor.rank}", flush=True)
    finally:
        executor.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("success", "rank-failure", "rank-exit"), required=True)
    parsed = parser.parse_args()
    _run(parsed.mode)


if __name__ == "__main__":
    main()
