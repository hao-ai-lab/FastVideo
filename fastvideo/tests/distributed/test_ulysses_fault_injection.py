# SPDX-License-Identifier: Apache-2.0
"""Rank-local Ulysses setup failures must produce a group-wide NCCL fallback."""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

SEED = 2026
TIMEOUT_SECONDS = 180
FAULT_CASES = [
    (2, 0, "capability"),
    (4, 3, "capability"),
    (2, 1, "configuration"),
    (4, 0, "configuration"),
    (2, 1, "allocation"),
    (4, 3, "allocation"),
    (2, 0, "post_registration"),
    (4, 3, "post_registration"),
    (2, 1, "post_dev_comm"),
    (4, 0, "post_dev_comm"),
    (2, 0, "teardown"),
    (4, 3, "teardown"),
    (2, 0, "capacity"),
    (4, 3, "capacity"),
    (2, 0, "capture"),
    (4, 3, "capture"),
    (2, 1, "layout"),
    (4, 0, "layout"),
    (2, 0, "lifecycle"),
    (4, 3, "lifecycle"),
]


def _worker() -> None:
    from fastvideo.distributed import (
        cleanup_dist_env_and_memory,
        maybe_init_distributed_environment_and_model_parallel,
    )
    from fastvideo.distributed.communication_op import sequence_model_parallel_all_to_all_4D
    from fastvideo.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase
    from fastvideo.distributed.parallel_state import get_sp_group

    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    fault_rank = int(os.environ["FASTVIDEO_ULYSSES_FAULT_RANK"])
    fault_stage = os.environ["FASTVIDEO_ULYSSES_FAULT_STAGE"]
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Construction capability is checked during distributed initialization, so
    # inject that failure before the communicator exists.
    if rank == fault_rank and fault_stage == "capability":
        from fastvideo.distributed.device_communicators.ulysses_a2a import UlyssesA2AHelper

        UlyssesA2AHelper._can_attempt = (  # type: ignore[method-assign]
            lambda self: (False, "injected capability failure"))
    elif rank == fault_rank and fault_stage == "configuration":
        os.environ["FASTVIDEO_ULYSSES_A2A"] = "off"

    maybe_init_distributed_environment_and_model_parallel(1, world)
    communicator = get_sp_group().device_communicator
    helper = communicator.ulysses_a2a

    try:
        if helper is None:
            if fault_stage not in ("capability", "configuration"):
                if rank == 0:
                    print("UNAVAILABLE helper not created", flush=True)
                return

        if helper is not None and rank == fault_rank and fault_stage == "allocation":
            helper._allocate = lambda nbytes: (_ for _ in ()).throw(
                RuntimeError(f"injected allocation failure for {nbytes} bytes"))
        elif helper is not None and rank == fault_rank and fault_stage == "post_registration":
            real_register = helper._register_window

            def _register_then_fail(handle: int) -> None:
                real_register(handle)
                raise RuntimeError("injected failure after registration")

            helper._register_window = _register_then_fail
        elif helper is not None and rank == fault_rank and fault_stage == "post_dev_comm":
            real_create = helper._create_dev_comm

            def _create_then_fail(handle: int) -> None:
                real_create(handle)
                raise RuntimeError("injected failure after device communicator creation")

            helper._create_dev_comm = _create_then_fail
        elif helper is not None and rank == fault_rank and fault_stage == "capacity":
            import fastvideo.distributed.device_communicators.ulysses_a2a as ulysses_module

            ulysses_module.MAX_WINDOW_BYTES = 1
        elif helper is not None and rank == fault_rank and fault_stage == "capture":
            torch.cuda.is_current_stream_capturing = lambda: True

        if fault_stage == "teardown":
            warmup = torch.randn(3, 32, 8, 64, device=device, dtype=torch.bfloat16)
            sequence_model_parallel_all_to_all_4D(warmup, 2, 1)
            assert helper is not None and helper._handle is not None
            if rank == fault_rank:
                real_dispose = helper._dispose

                def _dispose_then_fail(handle: int, *, synchronize: bool) -> None:
                    real_dispose(handle, synchronize=synchronize)
                    raise RuntimeError("injected failure after teardown")

                helper._dispose = _dispose_then_fail

        torch.manual_seed(SEED + rank)
        if fault_stage == "lifecycle":
            warmup = torch.randn(3, 32, 8, 64, device=device, dtype=torch.bfloat16)
            sequence_model_parallel_all_to_all_4D(warmup, 2, 1)
            assert helper is not None and helper._handle is not None
            if rank == fault_rank:
                helper._nbytes += 1

        if rank == fault_rank and fault_stage == "layout":
            storage = torch.randn(3, 64, 8, 128, device=device, dtype=torch.bfloat16)
            x = storage[..., ::2]
            assert not x.is_contiguous()
        else:
            sequence = 128 if fault_stage == "teardown" else 64
            x = torch.randn(3, sequence, 8, 64, device=device, dtype=torch.bfloat16)
        got = sequence_model_parallel_all_to_all_4D(x, 2, 1)
        expected = DeviceCommunicatorBase.all_to_all_4D(communicator, x, 2, 1)
        assert torch.equal(got, expected)
        if fault_stage == "layout":
            got_back = DeviceCommunicatorBase.all_to_all_4D(communicator, got.contiguous(), 1, 2)
        else:
            got_back = sequence_model_parallel_all_to_all_4D(got.contiguous(), 1, 2)
        assert torch.equal(got_back, x)

        armed = torch.tensor([int(helper is not None and helper._handle is not None)],
                             device=device, dtype=torch.int32)
        armed_min, armed_max = armed.clone(), armed.clone()
        dist.all_reduce(armed_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(armed_max, op=dist.ReduceOp.MAX)
        assert armed_min.item() == armed_max.item() == 0
        print(f"RANK_DONE rank={rank} armed=False", flush=True)
        if rank == 0:
            print(f"ALL_RANKS_COMPLETED world={world} stage={fault_stage}", flush=True)
    finally:
        cleanup_dist_env_and_memory()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.parametrize("world,fault_rank,fault_stage", FAULT_CASES)
def test_rank_local_setup_failure_falls_back_group_wide(world: int, fault_rank: int,
                                                        fault_stage: str) -> None:
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs at least {world} CUDA devices")
    environment = dict(
        os.environ,
        FASTVIDEO_ULYSSES_A2A="auto",
        FASTVIDEO_ULYSSES_FAULT_RANK=str(fault_rank),
        FASTVIDEO_ULYSSES_FAULT_STAGE=fault_stage,
    )
    try:
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={world}",
                f"--master_port={_free_port()}",
                str(Path(__file__).resolve()),
                "--worker",
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"rank {fault_rank} {fault_stage} failure deadlocked the group")

    if "UNAVAILABLE" in process.stdout:
        pytest.skip(process.stdout.strip())
    assert process.returncode == 0 and "ALL_RANKS_COMPLETED" in process.stdout, (
        f"stdout:\n{process.stdout}\nstderr:\n{process.stderr[-6000:]}")
    armed = dict(re.findall(r"RANK_DONE rank=(\d+) armed=(True|False)", process.stdout))
    assert len(armed) == world and set(armed.values()) == {"False"}


if __name__ == "__main__" and "--worker" in sys.argv:
    _worker()
