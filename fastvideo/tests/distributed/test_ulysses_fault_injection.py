# SPDX-License-Identifier: Apache-2.0
"""A rank-local failure must not strand its peers.

The fused all-to-all's first instruction is a barrier across every rank. So if
one rank decides not to use the fused path while the others do, the others wait
for an arrival that never comes and the job hangs -- no exception, no output,
until the NCCL watchdog fires minutes later. That is the worst failure mode this
feature can produce, and it is invisible to a correctness test: every existing
test passes because every rank agrees.

This asserts the property directly. One rank is forced to decline the fused path
while the others would accept it; the run must still finish, with correct
results, on the NCCL path. A hang fails the test via timeout rather than wedging
the suite.

The realistic triggers are unexciting and entirely plausible: a partial install
where ``import flashinfer`` fails on one node, one rank short on memory when the
staging buffer is allocated, or a JIT cache race. The failure they produce is
not unexciting at all.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch

SEED = 2026
TIMEOUT_S = 120


def _worker() -> None:
    """Child process: decline the fused path on one rank, then do a collective."""
    from fastvideo.distributed import (cleanup_dist_env_and_memory,
                                       maybe_init_distributed_environment_and_model_parallel)
    from fastvideo.distributed.communication_op import sequence_model_parallel_all_to_all_4D
    from fastvideo.distributed.device_communicators.base_device_communicator import (
        DeviceCommunicatorBase)
    from fastvideo.distributed.parallel_state import get_sp_group, get_sp_world_size

    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    fault_rank = int(os.environ.get("FASTVIDEO_ULYSSES_FAULT_RANK", "-1"))

    maybe_init_distributed_environment_and_model_parallel(1, world)
    w = get_sp_world_size()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    comm = get_sp_group().device_communicator
    helper = comm.ulysses_a2a

    # Inject a realistic rank-local failure on this rank only, of the kind a
    # partial install or a flaky shared mount produces: the backend is simply
    # not usable here.
    #
    # Injected at _can_attempt because that is the decision the ranks vote on,
    # and it is deliberately evaluated before anything is imported or allocated.
    # Injecting later -- at _build, say -- would not exercise the fix at all: a
    # rank that fails after its peers have already entered the backend's own
    # collective protocol has stranded them regardless of what it does next.
    # Injecting at a specific module name would instead couple the test to
    # whichever backend is wired up today.
    if rank == fault_rank:
        from fastvideo.distributed.device_communicators.ulysses_a2a import UlyssesA2AHelper

        UlyssesA2AHelper._can_attempt = (  # type: ignore[method-assign]
            lambda self: (False, f"injected fault on rank {rank} (test)"))

    try:
        torch.manual_seed(SEED + rank)
        x = torch.randn(3, 64, 8, 64, device=device, dtype=torch.bfloat16)

        got = sequence_model_parallel_all_to_all_4D(x, scatter_dim=2, gather_dim=1)
        want = DeviceCommunicatorBase.all_to_all_4D(comm, x, 2, 1)
        assert torch.equal(got, want), f"rank {rank}: result diverged from the NCCL path"

        # Round-trip too: the return leg must agree on the same backend.
        back = sequence_model_parallel_all_to_all_4D(got.contiguous(), scatter_dim=1, gather_dim=2)
        assert torch.equal(back, x), f"rank {rank}: round-trip failed"

        armed = helper is not None and helper._handle is not None
        print(f"RANKDONE rank={rank} armed={armed}", flush=True)
        if rank == 0:
            print(f"ALL_RANKS_COMPLETED world={w} fault_rank={fault_rank}", flush=True)
    finally:
        cleanup_dist_env_and_memory()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run(fault_rank: int, world: int) -> subprocess.CompletedProcess:
    env = dict(os.environ,
               FASTVIDEO_ULYSSES_A2A="auto",
               FASTVIDEO_ULYSSES_FAULT_RANK=str(fault_rank))
    return subprocess.run(
        [
            sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={world}",
            f"--master_port={_free_port()}",
            str(Path(__file__).resolve()), "--worker",
        ],
        env=env, capture_output=True, text=True, timeout=TIMEOUT_S,
    )


@pytest.mark.parametrize("world,fault_rank", [(2, 0), (2, 1), (4, 0), (4, 3)])
def test_one_rank_declining_does_not_hang(world: int, fault_rank: int) -> None:
    """With one rank unable to arm, every rank must still finish, on NCCL."""
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs >= {world} GPUs")
    try:
        proc = _run(fault_rank, world)
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"HANG: rank {fault_rank} declined the fused path and the group deadlocked "
            f"(no completion within {TIMEOUT_S}s). Ranks must agree on the backend "
            f"group-wide before any rank enters the kernel.")

    assert "ALL_RANKS_COMPLETED" in proc.stdout, (
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr[-4000:]}")

    # Every rank must land on the SAME backend. A split is the bug, even when it
    # happens not to hang for a given shape. Parsed with a regex because ranks
    # write concurrently and their lines can interleave without a newline.
    armed = dict(re.findall(r"RANKDONE rank=(\d+) armed=(True|False)", proc.stdout))
    assert len(armed) == world, (
        f"expected {world} ranks to report, got {armed}\n{proc.stdout}")
    assert set(armed.values()) == {"False"}, (
        f"ranks disagreed on the backend, or one stayed armed while a peer fell "
        f"back: {armed}")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _worker()
