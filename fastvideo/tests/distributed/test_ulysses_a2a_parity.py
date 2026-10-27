# SPDX-License-Identifier: Apache-2.0
"""Parity and stress coverage for the fused Ulysses all-to-all."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

SEED = 2026
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


def _all_to_all_single(inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    world = len(inputs)
    return [torch.cat([inputs[src].chunk(world, dim=0)[rank] for src in range(world)], dim=0)
            for rank in range(world)]


def _fastvideo_scatter(xs: list[torch.Tensor], world: int) -> list[torch.Tensor]:
    local_heads = xs[0].shape[2] // world
    recvs = _all_to_all_single([x.transpose(0, 2).contiguous() for x in xs])
    return [torch.cat(output.split(local_heads), dim=1).transpose(0, 2).contiguous()
            for output in recvs]


def _fastvideo_gather(ys: list[torch.Tensor], world: int) -> list[torch.Tensor]:
    sends = []
    for y in ys:
        batch, sequence, local_heads, head_dim = y.shape
        local_sequence = sequence // world
        transposed = y.transpose(0, 2).contiguous()
        sends.append(transposed.reshape(local_heads, world, local_sequence, batch, head_dim).transpose(
            0, 1).reshape(local_heads * world, local_sequence, batch, head_dim).contiguous())
    return [output.transpose(0, 2).contiguous() for output in _all_to_all_single(sends)]


def _kernel_scatter(xs: list[torch.Tensor], world: int) -> list[torch.Tensor]:
    batch, local_sequence, heads, head_dim = xs[0].shape
    local_heads = heads // world
    outputs = []
    for rank in range(world):
        output = torch.empty(batch, local_sequence * world, local_heads, head_dim, dtype=xs[0].dtype)
        for src in range(world):
            output[:, src * local_sequence:(src + 1) * local_sequence] = xs[src][
                :, :, rank * local_heads:(rank + 1) * local_heads]
        outputs.append(output)
    return outputs


def _kernel_gather(ys: list[torch.Tensor], world: int) -> list[torch.Tensor]:
    batch, global_sequence, local_heads, head_dim = ys[0].shape
    local_sequence = global_sequence // world
    outputs = []
    for dst in range(world):
        output = torch.empty(batch, local_sequence, local_heads * world, head_dim, dtype=ys[0].dtype)
        for src in range(world):
            output[:, :, src * local_heads:(src + 1) * local_heads] = ys[src][
                :, dst * local_sequence:(dst + 1) * local_sequence]
        outputs.append(output)
    return outputs


@pytest.mark.parametrize("world,batch,local_sequence,heads,head_dim", CPU_CASES)
def test_layout_equivalence_cpu(world: int, batch: int, local_sequence: int, heads: int,
                                head_dim: int) -> None:
    """The kernel index formula must exactly match the existing NCCL layout."""
    torch.manual_seed(SEED)
    xs = [torch.randn(batch, local_sequence, heads, head_dim) for _ in range(world)]

    reference = _fastvideo_scatter(xs, world)
    fused = _kernel_scatter(xs, world)
    assert all(torch.equal(left, right) for left, right in zip(reference, fused))

    reference_back = _fastvideo_gather(fused, world)
    fused_back = _kernel_gather(fused, world)
    assert all(torch.equal(left, right) for left, right in zip(reference_back, fused_back))
    assert all(torch.equal(original, round_trip) for original, round_trip in zip(xs, fused_back))


def test_fullgraph_compile_declines_before_distributed_state() -> None:
    """Compiled regions must route to the existing compiler-visible NCCL op."""
    from fastvideo.distributed.device_communicators.ulysses_a2a import UlyssesA2AHelper

    helper = UlyssesA2AHelper(object(), object(), 4, torch.device("cuda:0"), object())
    with patch("torch.compiler.is_compiling", return_value=True):
        assert helper.try_all_to_all_4D(torch.empty(1), 2, 1) is None


def _check_shape(batch: int, local_sequence: int, heads: int, head_dim: int,
                 dtype: torch.dtype, world: int, device: torch.device) -> None:
    from fastvideo.distributed.communication_op import sequence_model_parallel_all_to_all_4D
    from fastvideo.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase
    from fastvideo.distributed.parallel_state import get_sp_group

    communicator = get_sp_group().device_communicator
    tag = f"{(batch, local_sequence, heads, head_dim)} {dtype}"
    x = torch.randn(batch, local_sequence, heads, head_dim, device=device, dtype=dtype)

    got = sequence_model_parallel_all_to_all_4D(x, 2, 1)
    expected = DeviceCommunicatorBase.all_to_all_4D(communicator, x, 2, 1)
    assert torch.equal(got, expected), f"scatter parity failed at {tag}"

    got_back = sequence_model_parallel_all_to_all_4D(got.contiguous(), 1, 2)
    expected_back = DeviceCommunicatorBase.all_to_all_4D(communicator, got.contiguous(), 1, 2)
    assert torch.equal(got_back, expected_back), f"gather parity failed at {tag}"
    assert torch.equal(got_back, x), f"round-trip failed at {tag}"

    x_grad = x.clone().requires_grad_(True)
    grad = torch.randn(batch, local_sequence * world, heads // world, head_dim,
                       device=device, dtype=dtype)
    sequence_model_parallel_all_to_all_4D(x_grad, 2, 1).backward(grad)
    expected_grad = DeviceCommunicatorBase.all_to_all_4D(communicator, grad.contiguous(), 1, 2)
    assert torch.equal(x_grad.grad, expected_grad), f"scatter backward failed at {tag}"

    y_grad = got.detach().clone().requires_grad_(True)
    reverse_grad = torch.randn_like(x)
    sequence_model_parallel_all_to_all_4D(y_grad, 1, 2).backward(reverse_grad)
    expected_reverse = DeviceCommunicatorBase.all_to_all_4D(communicator,
                                                             reverse_grad.contiguous(), 2, 1)
    assert torch.equal(y_grad.grad, expected_reverse), f"gather backward failed at {tag}"


def _worker() -> None:
    from fastvideo.distributed import (
        cleanup_dist_env_and_memory,
        maybe_init_distributed_environment_and_model_parallel,
    )
    from fastvideo.distributed.communication_op import sequence_model_parallel_all_to_all_4D
    from fastvideo.distributed.parallel_state import get_sp_group

    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    maybe_init_distributed_environment_and_model_parallel(1, world)
    communicator = get_sp_group().device_communicator
    helper = communicator.ulysses_a2a
    torch.manual_seed(SEED + rank)

    try:
        for batch, local_sequence, heads, head_dim in [
                (3, 32, 8, 64), (3, 64, 40, 128), (1, 48, 56, 128)]:
            if heads % world:
                continue
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                _check_shape(batch, local_sequence, heads, head_dim, dtype, world, device)

        # Grow once, then keep all 36 CTAs crossing two barriers for long enough
        # to catch shared-generation races or teardown-before-completion bugs.
        stress = torch.randn(3, 64, 56, 128, device=device, dtype=torch.bfloat16)
        for _ in range(100):
            scattered = sequence_model_parallel_all_to_all_4D(stress, 2, 1)
            stress_back = sequence_model_parallel_all_to_all_4D(scattered.contiguous(), 1, 2)
            assert torch.equal(stress_back, stress)

        engaged = torch.tensor([int(helper is not None and helper._handle is not None)],
                               device=device, dtype=torch.int32)
        engaged_min, engaged_max = engaged.clone(), engaged.clone()
        dist.all_reduce(engaged_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(engaged_max, op=dist.ReduceOp.MAX)
        assert engaged_min.item() == engaged_max.item(), "ranks disagreed on fused engagement"

        capable = False
        try:
            from fastvideo_kernel import comm_ops

            pynccl = communicator.pynccl_comm
            comm = pynccl.comm
            comm_ptr = int(getattr(comm, "value", comm))
            capable = (comm_ops.is_available()
                       and comm_ops.lsa_covers_group(comm_ptr, world))
        except Exception:  # noqa: BLE001 - a missing optional backend is a valid fallback
            pass
        capable_vote = torch.tensor([int(capable)], device=device, dtype=torch.int32)
        dist.all_reduce(capable_vote, op=dist.ReduceOp.MIN)
        if capable_vote.item():
            assert engaged_min.item() == 1, "capable topology did not arm the fused backend"
        if rank == 0:
            reason = helper._disabled_reason if helper is not None else "helper not created"
            print(f"PARITY_OK fused_engaged={bool(engaged.item())} reason={reason}", flush=True)
    finally:
        cleanup_dist_env_and_memory()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.parametrize("world", [2, 4, 8])
def test_fused_matches_nccl_gpu(world: int) -> None:
    """Forward/backward, growth and a 100-round multi-CTA stress stay exact."""
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs at least {world} CUDA devices")
    environment = dict(os.environ, FASTVIDEO_ULYSSES_A2A="auto")
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
        timeout=1800,
    )
    assert process.returncode == 0 and "PARITY_OK" in process.stdout, (
        f"stdout:\n{process.stdout}\nstderr:\n{process.stderr[-6000:]}")


if __name__ == "__main__" and "--worker" in sys.argv:
    _worker()
