# SPDX-License-Identifier: Apache-2.0
"""Parity between the fused Ulysses all-to-all and the NCCL path.

The fused kernel is only a drop-in replacement if it produces exactly the same
layout as ``DistributedAutograd.AllToAll4D``. Both are supposed to compute

    scatter (dims 2->1):  y_r[b, j*S_local + s, hl, d] = x_j[b, s, r*H_local + hl, d]
    gather  (dims 1->2):  the exact inverse

so this asserts byte equality (they are pure data movement -- no arithmetic, so
"close enough" is not the bar), in both directions and through backward.

Two tiers, so the important half runs anywhere:

* ``test_layout_equivalence_cpu`` needs no GPU and no distributed init. It
  simulates all W ranks in one process and checks FastVideo's transpose/split/cat
  against the kernel's index formula. This is what pins down the layout claim.
* ``test_fused_matches_nccl_gpu`` runs the real collectives under torchrun and
  additionally reports whether the fused backend engaged at all -- on hardware
  that is not a single-node NVLink mesh it will not, and the test then only
  confirms the fallback is correct.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch

SEED = 2026

# (W, B, S_local, H, D). B=3 mirrors the torch.cat([q, k, v], dim=0) that
# DistributedAttention issues, so the batch dim is exercised the way the model
# actually uses it. H=40 is Wan-14B, H=56 is MiniMax-H3.
CPU_CASES = [
    (2, 1, 4, 4, 2),
    (2, 3, 5, 8, 4),
    (4, 1, 3, 8, 2),
    (4, 3, 7, 40, 16),
    (4, 3, 6, 56, 16),
    (6, 2, 3, 42, 8),
    (8, 3, 2, 56, 8),
    (8, 1, 5, 40, 4),
]


# --------------------------------------------------------------------------
# Tier 1: layout equivalence, no GPU required
# --------------------------------------------------------------------------
def _all_to_all_single(inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    """``dist.all_to_all_single`` over dim 0, for every rank at once.

    By definition rank r's j-th output chunk is rank j's r-th input chunk.
    """
    w = len(inputs)
    return [torch.cat([inputs[j].chunk(w, dim=0)[r] for j in range(w)], dim=0) for r in range(w)]


def _fastvideo_scatter(xs: list[torch.Tensor], w: int) -> list[torch.Tensor]:
    """Mirror of DistributedAutograd.AllToAll4D scatter_dim=2, gather_dim=1."""
    shard_hn = xs[0].shape[2] // w
    recvs = _all_to_all_single([x.transpose(0, 2).contiguous() for x in xs])
    return [torch.cat(o.split(shard_hn), dim=1).transpose(0, 2).contiguous() for o in recvs]


def _fastvideo_gather(ys: list[torch.Tensor], w: int) -> list[torch.Tensor]:
    """Mirror of DistributedAutograd.AllToAll4D scatter_dim=1, gather_dim=2."""
    sends = []
    for y in ys:
        bs, seqlen, shard_hn, hd = y.shape
        shard_seqlen = seqlen // w
        t = y.transpose(0, 2).contiguous()
        sends.append(t.reshape(shard_hn, w, shard_seqlen, bs, hd).transpose(0, 1).reshape(
            shard_hn * w, shard_seqlen, bs, hd).contiguous())
    return [o.transpose(0, 2).contiguous() for o in _all_to_all_single(sends)]


def _kernel_scatter(xs: list[torch.Tensor], w: int) -> list[torch.Tensor]:
    """The index formula in flashinfer's ulysses_all_to_all.cuh, mode 0."""
    B, S_local, H, D = xs[0].shape
    H_local = H // w
    outs = []
    for r in range(w):
        y = torch.empty(B, S_local * w, H_local, D, dtype=xs[0].dtype)
        for j in range(w):
            y[:, j * S_local:(j + 1) * S_local] = xs[j][:, :, r * H_local:(r + 1) * H_local]
        outs.append(y)
    return outs


def _kernel_gather(ys: list[torch.Tensor], w: int) -> list[torch.Tensor]:
    """The index formula in flashinfer's ulysses_all_to_all.cuh, mode 1."""
    B, S_global, H_local, D = ys[0].shape
    S_local = S_global // w
    outs = []
    for j in range(w):
        o = torch.empty(B, S_local, H_local * w, D, dtype=ys[0].dtype)
        for r in range(w):
            o[:, :, r * H_local:(r + 1) * H_local] = ys[r][:, j * S_local:(j + 1) * S_local]
        outs.append(o)
    return outs


@pytest.mark.parametrize("w,B,S_local,H,D", CPU_CASES)
def test_layout_equivalence_cpu(w: int, B: int, S_local: int, H: int, D: int) -> None:
    """FastVideo's permute-based a2a == the fused kernel's index formula."""
    torch.manual_seed(SEED)
    xs = [torch.randn(B, S_local, H, D) for _ in range(w)]

    fv, kern = _fastvideo_scatter(xs, w), _kernel_scatter(xs, w)
    for r, (a, b) in enumerate(zip(fv, kern)):
        assert torch.equal(a, b), f"scatter mismatch on rank {r}"

    fv_g, kern_g = _fastvideo_gather(kern, w), _kernel_gather(kern, w)
    for r, (a, b) in enumerate(zip(fv_g, kern_g)):
        assert torch.equal(a, b), f"gather mismatch on rank {r}"

    # gather(scatter(x)) must be the identity
    for r, (a, b) in enumerate(zip(xs, kern_g)):
        assert torch.equal(a, b), f"round-trip mismatch on rank {r}"


# --------------------------------------------------------------------------
# Tier 2: real collectives under torchrun
# --------------------------------------------------------------------------
def _worker() -> None:
    """Body of the torchrun'd child process."""
    from fastvideo.distributed import (cleanup_dist_env_and_memory,
                                       maybe_init_distributed_environment_and_model_parallel)
    from fastvideo.distributed.communication_op import sequence_model_parallel_all_to_all_4D
    from fastvideo.distributed.parallel_state import get_sp_group, get_sp_world_size

    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    maybe_init_distributed_environment_and_model_parallel(1, world)
    w = get_sp_world_size()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    helper = get_sp_group().device_communicator.ulysses_a2a
    torch.manual_seed(SEED + rank)

    try:
        for B, S_local, H, D in [(3, 32, 8, 64), (3, 64, 40, 128)]:
            if H % w:
                continue
            x = torch.randn(B, S_local, H, D, device=device, dtype=torch.bfloat16)

            # Fused (or fallback, if the topology is not eligible).
            got = sequence_model_parallel_all_to_all_4D(x, scatter_dim=2, gather_dim=1)

            # Reference: the inherited NCCL implementation, reached directly.
            from fastvideo.distributed.device_communicators.base_device_communicator import (
                DeviceCommunicatorBase)
            comm = get_sp_group().device_communicator
            want = DeviceCommunicatorBase.all_to_all_4D(comm, x, 2, 1)
            assert torch.equal(got, want), f"scatter parity failed at {(B, S_local, H, D)}"

            y = got.contiguous()
            got_g = sequence_model_parallel_all_to_all_4D(y, scatter_dim=1, gather_dim=2)
            want_g = DeviceCommunicatorBase.all_to_all_4D(comm, y, 1, 2)
            assert torch.equal(got_g, want_g), f"gather parity failed at {(B, S_local, H, D)}"
            assert torch.equal(got_g, x), f"round-trip failed at {(B, S_local, H, D)}"

            # Backward: scatter's vjp is a gather, with no scaling.
            xg = x.clone().requires_grad_(True)
            g = torch.randn(B, S_local * w, H // w, D, device=device, dtype=torch.bfloat16)
            sequence_model_parallel_all_to_all_4D(xg, 2, 1).backward(g)
            want_grad = DeviceCommunicatorBase.all_to_all_4D(comm, g.contiguous(), 1, 2)
            assert torch.equal(xg.grad, want_grad), f"backward parity failed at {(B, S_local, H, D)}"

        # A dtype or capacity change must rebuild the communicator, not
        # permanently disable the fused path: the first operand a helper sees is
        # not necessarily representative (the SP warmup used to arm it on tiny
        # bf16 dummies, which then locked out an fp32 model entirely).
        for dtype in (torch.float32, torch.bfloat16):
            big = torch.randn(3, 128, 8, 64, device=device, dtype=dtype)
            got = sequence_model_parallel_all_to_all_4D(big, scatter_dim=2, gather_dim=1)
            from fastvideo.distributed.device_communicators.base_device_communicator import (
                DeviceCommunicatorBase)
            comm = get_sp_group().device_communicator
            want = DeviceCommunicatorBase.all_to_all_4D(comm, big, 2, 1)
            assert torch.equal(got, want), f"parity failed after switching to {dtype}"
            if helper is not None and helper._disabled_reason is None:
                assert helper._dtype == dtype, (
                    f"helper did not rebuild for {dtype} (still {helper._dtype})")

        if rank == 0:
            armed = helper is not None and helper._comm is not None
            reason = helper._disabled_reason if helper is not None else "helper not created"
            print(f"PARITY_OK fused_engaged={armed} reason={reason}", flush=True)
    finally:
        cleanup_dist_env_and_memory()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs >= 2 GPUs")
def test_fused_matches_nccl_gpu() -> None:
    """The seam produces byte-identical results to the NCCL path it replaces."""
    env = dict(os.environ, FASTVIDEO_ULYSSES_A2A="auto")
    proc = subprocess.run(
        [
            sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2",
            f"--master_port={_free_port()}",
            str(Path(__file__).resolve()), "--worker",
        ],
        env=env, capture_output=True, text=True, timeout=1800,
    )
    assert "PARITY_OK" in proc.stdout, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr[-4000:]}"
    print([ln for ln in proc.stdout.splitlines() if "PARITY_OK" in ln][0])


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _worker()
