# SPDX-License-Identifier: Apache-2.0
"""Distributed correctness for the vendored pure Ring Attention kernel.

Ring Attention splits the sequence across ranks and streams K/V blocks
around the ring, combining partial softmax results via the log-sum-exp
trick. This test checks that end-to-end mathematical property directly
against ``fastvideo.attention.ring.ring_flash_attn_func``: every rank shards
Q/K/V from the same global tensors, runs the ring kernel, and the gathered
result must match plain (single-rank) FlashAttention run over the full
sequence.

This does not go through ``DistributedAttention``/``FastVideoArgs`` — it
targets the vendored kernel in isolation so a failure here points straight
at the kernel rather than at FastVideo's integration layer (which is
covered separately by the config and RoPE-slicing tests in
``fastvideo/tests/attention/``).
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
from pathlib import Path

import pytest
import torch

RING_WORLD_SIZE = 2
SEED = 2026


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_worker(output_path: Path) -> None:
    import torch.distributed as dist

    from fastvideo.attention.ring import ring_flash_attn_func
    from fastvideo.distributed import (
        cleanup_dist_env_and_memory,
        get_ring_group,
        maybe_init_distributed_environment_and_model_parallel,
    )
    from flash_attn import flash_attn_func

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    try:
        maybe_init_distributed_environment_and_model_parallel(1, RING_WORLD_SIZE, ring_size=RING_WORLD_SIZE)

        batch, global_seq_len, heads, head_size = 1, 256, 4, 64
        local_seq_len = global_seq_len // RING_WORLD_SIZE

        generator = torch.Generator(device="cpu").manual_seed(SEED)
        q_global = torch.randn(batch, global_seq_len, heads, head_size, generator=generator,
                               dtype=torch.bfloat16).to(device)
        k_global = torch.randn(batch, global_seq_len, heads, head_size, generator=generator,
                               dtype=torch.bfloat16).to(device)
        v_global = torch.randn(batch, global_seq_len, heads, head_size, generator=generator,
                               dtype=torch.bfloat16).to(device)
        # every rank must see identical global tensors
        dist.broadcast(q_global, src=0)
        dist.broadcast(k_global, src=0)
        dist.broadcast(v_global, src=0)

        start = rank * local_seq_len
        end = start + local_seq_len
        q_local = q_global[:, start:end].contiguous()
        k_local = k_global[:, start:end].contiguous()
        v_local = v_global[:, start:end].contiguous()

        softmax_scale = head_size**-0.5
        ring_out_local = ring_flash_attn_func(
            q_local,
            k_local,
            v_local,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=False,
            group=get_ring_group(),
        )

        gathered = [torch.empty_like(ring_out_local) for _ in range(RING_WORLD_SIZE)]
        dist.all_gather(gathered, ring_out_local)
        ring_out = torch.cat(gathered, dim=1)

        if rank == 0:
            reference = flash_attn_func(
                q_global,
                k_global,
                v_global,
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=False,
            )
            torch.save({"ring_out": ring_out.cpu(), "reference": reference.cpu()}, output_path)

        dist.barrier()
    finally:
        cleanup_dist_env_and_memory()


def _run_torchrun(script_path: Path, nproc_per_node: int, output_path: Path) -> None:
    cmd = [
        "torchrun",
        "--nnodes",
        "1",
        "--nproc_per_node",
        str(nproc_per_node),
        "--master_port",
        str(_free_port()),
        str(script_path),
        "--ring-worker",
        "--output",
        str(output_path),
    ]
    process = subprocess.run(cmd, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"Ring worker failed with code {process.returncode}\n"
                          f"STDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}")


def test_ring_attention_matches_full_attention(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("This test requires CUDA.")
    if torch.cuda.device_count() < RING_WORLD_SIZE:
        pytest.skip(f"This test requires at least {RING_WORLD_SIZE} CUDA devices.")

    script_path = Path(__file__).resolve()
    output_path = tmp_path / "ring_vs_full.pt"
    _run_torchrun(script_path, nproc_per_node=RING_WORLD_SIZE, output_path=output_path)

    saved = torch.load(output_path, map_location="cpu")
    ring_out, reference = saved["ring_out"], saved["reference"]

    assert torch.isfinite(ring_out).all()
    assert torch.isfinite(reference).all()
    torch.testing.assert_close(ring_out.float(), reference.float(), rtol=3e-2, atol=3e-2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ring-worker", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


# pytest -sv fastvideo/tests/distributed/test_ring_attention.py
if __name__ == "__main__":
    args = _parse_args()
    if not args.ring_worker:
        raise SystemExit("This module is intended to be run by pytest.")
    if args.output is None:
        raise SystemExit("--output is required in worker mode.")
    _run_worker(Path(args.output))
