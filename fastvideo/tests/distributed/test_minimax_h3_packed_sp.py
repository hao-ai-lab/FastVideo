# SPDX-License-Identifier: Apache-2.0
"""Real world-4 contract coverage for MiniMax-H3 packed sequence parallelism."""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

SP_WORLD_SIZE = 4


class _TestGroup:

    def __init__(self, world_size: int, device: torch.device) -> None:
        self.world_size = world_size
        self.device = device
        self.device_group = dist.group.WORLD
        self.unique_name = "minimax_h3_packed_sp_test"


class _QKVOracleAttentionImpl(nn.Module):

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, metadata: object) -> torch.Tensor:
        del metadata
        return q + 2 * k + 3 * v

    def postprocess_output(self, output: torch.Tensor, metadata: object) -> torch.Tensor:
        del metadata
        return output


def _reference_pack(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, world_size: int) -> torch.Tensor:
    rows, heads, head_dim = q.shape
    heads_local = heads // world_size
    output = torch.empty(world_size,
                         rows,
                         heads_local,
                         3 * head_dim,
                         device=q.device,
                         dtype=q.dtype)
    for index, tensor in enumerate((q, k, v)):
        shard = tensor.reshape(rows, world_size, heads_local, head_dim).permute(1, 0, 2, 3)
        output[..., index * head_dim:(index + 1) * head_dim].copy_(shard)
    return output


def _reference_merge(output: torch.Tensor) -> torch.Tensor:
    world, rows, heads_local, head_dim = output.shape
    return output.permute(1, 0, 2, 3).contiguous().reshape(rows, world * heads_local, head_dim)


def _exact_cuda_route(world: int) -> bool:
    strict_cuda = os.environ.get("FASTVIDEO_MINIMAX_H3_PACKED_SP_STRICT_CUDA", "0") == "1"
    visible_cuda_devices = torch.cuda.device_count()
    if strict_cuda and visible_cuda_devices < world:
        raise RuntimeError(
            "strict MiniMax-H3 packed-SP preflight requires one visible CUDA device per rank; "
            f"world={world}, visible={visible_cuda_devices}")
    return visible_cuda_devices >= world


def _worker() -> None:
    from fastvideo.attention.layer import DistributedAttention
    from fastvideo.distributed import communication_op, parallel_state
    from fastvideo.forward_context import set_forward_context
    from fastvideo.models.dits.minimax_h3_fusions import relayout

    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    exact_cuda_route = _exact_cuda_route(world)
    if exact_cuda_route:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dtype = torch.bfloat16
        dist.init_process_group(backend="nccl", device_id=device)
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        dist.init_process_group(backend="gloo")

    group = _TestGroup(world, device)
    parallel_state._register_group(group)
    attention = DistributedAttention.__new__(DistributedAttention)
    nn.Module.__init__(attention)
    attention.attn_impl = _QKVOracleAttentionImpl()
    attention.head_size = 16
    attention.packed_qkv_relayout = True
    attention._compile_forward_enabled = True

    rows_local, heads, head_dim = 7, 8, 16
    values = torch.arange(rows_local * heads * head_dim, device=device, dtype=torch.float32)
    q = (values.reshape(1, rows_local, heads, head_dim) + rank * 10_000).to(dtype)
    k = q + 100
    v = q + 200
    semantic_rows = world * rows_local - 3
    expected = q + 2 * k + 3 * v
    if rank == world - 1:
        expected[:, -3:] = 0

    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("fastvideo.attention.layer.get_sp_world_size", return_value=world))
            stack.enter_context(patch("fastvideo.attention.layer.get_sp_parallel_rank", return_value=rank))
            stack.enter_context(patch.object(communication_op, "get_sp_group", return_value=group))
            if not exact_cuda_route:
                # The production relayout kernels are independently compiled
                # and bit-exact-tested on CUDA. Gloo lets every development
                # machine exercise the same two real world-4 collectives and
                # rank ordering without pretending Triton runs on CPU.
                stack.enter_context(patch.object(relayout, "pack_qkv_destination_major", _reference_pack))
                stack.enter_context(patch.object(relayout, "merge_heads", _reference_merge))
            stack.enter_context(torch.inference_mode())
            stack.enter_context(set_forward_context(current_timestep=0, attn_metadata=None))
            output, replicated = attention(q, k, v, original_seq_len=semantic_rows)

        assert replicated is None
        assert torch.equal(output, expected), f"rank {rank} packed scatter/gather round-trip differed"

        with patch.object(communication_op, "get_sp_group", return_value=group):
            with torch.no_grad(), pytest.raises(ValueError, match="leading dimension"):
                communication_op.sequence_model_parallel_direct_all_to_all(torch.empty(world + 1, 2, device=device))
            with pytest.raises(RuntimeError, match="inference-only"):
                communication_op.sequence_model_parallel_direct_all_to_all(
                    torch.empty(world, 2, device=device, requires_grad=True))

        dist.barrier()
        if rank == 0:
            mode = "cuda-production" if exact_cuda_route else "gloo-contract"
            print(f"MINIMAX_H3_PACKED_SP_OK world={world} mode={mode}", flush=True)
    finally:
        dist.destroy_process_group()


def test_minimax_h3_packed_sp_world4_collective_contract() -> None:
    environment = dict(os.environ, OMP_NUM_THREADS="1")
    launcher = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={SP_WORLD_SIZE}",
    ]
    inherited_master_port = os.environ.get("MASTER_PORT")
    if inherited_master_port:
        launcher.append(f"--master_port={inherited_master_port}")
    else:
        # Local invocations have no scheduler-assigned port. Let torchrun own
        # discovery instead of racing a bind-and-close port probe.
        launcher.append("--standalone")
    launcher.extend([
        str(Path(__file__).resolve()),
        "--worker",
    ])
    process = subprocess.run(
        launcher,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )
    strict_cuda = os.environ.get("FASTVIDEO_MINIMAX_H3_PACKED_SP_STRICT_CUDA", "0") == "1"
    expected_mode = "mode=cuda-production" if strict_cuda else "MINIMAX_H3_PACKED_SP_OK"
    assert process.returncode == 0 and expected_mode in process.stdout, (
        f"stdout:\n{process.stdout[-6000:]}\nstderr:\n{process.stderr[-6000:]}")


def test_minimax_h3_packed_sp_strict_mode_rejects_insufficient_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTVIDEO_MINIMAX_H3_PACKED_SP_STRICT_CUDA", "1")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: SP_WORLD_SIZE - 1)
    with pytest.raises(RuntimeError, match="one visible CUDA device per rank"):
        _exact_cuda_route(SP_WORLD_SIZE)


if __name__ == "__main__" and "--worker" in sys.argv:
    _worker()
