# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the vendored Ring Attention implementation.

This file contains three levels of tests:

1. Single-GPU Ring kernel fallback:
   Ring Attention with a one-rank process group must match plain
   FlashAttention.

2. Single-GPU blockwise softmax merge:
   Attention computed over multiple KV blocks and merged through the
   log-sum-exp update must match attention over the concatenated KV sequence.

3. Multi-GPU Ring Attention:
   When at least two GPUs are available, Q/K/V are sequence-sharded across
   ranks and the gathered Ring Attention output must match full-sequence
   FlashAttention.

The first two tests can run on a single GPU. The final test validates actual
KV communication and requires at least two GPUs.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
from pathlib import Path

import pytest
import torch

SEED = 2026
MULTI_GPU_RING_WORLD_SIZE = 2

BATCH_SIZE = 1
GLOBAL_SEQ_LEN = 256
NUM_HEADS = 4
HEAD_SIZE = 64

RTOL = 3e-2
ATOL = 3e-2


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _make_qkv(
    *,
    batch_size: int,
    sequence_length: int,
    num_heads: int,
    head_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create deterministic Q/K/V tensors."""

    generator = torch.Generator(device="cpu").manual_seed(SEED)

    shape = (
        batch_size,
        sequence_length,
        num_heads,
        head_size,
    )

    q = torch.randn(
        shape,
        generator=generator,
        dtype=dtype,
    ).to(device)

    k = torch.randn(
        shape,
        generator=generator,
        dtype=dtype,
    ).to(device)

    v = torch.randn(
        shape,
        generator=generator,
        dtype=dtype,
    ).to(device)

    return q, k, v


def _assert_attention_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()

    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        rtol=RTOL,
        atol=ATOL,
    )


def test_ring_attention_world_size_one_matches_flash_attention() -> None:
    """A one-rank Ring must reduce to ordinary FlashAttention."""

    if not torch.cuda.is_available():
        pytest.skip("This test requires CUDA.")

    from flash_attn import flash_attn_func

    from fastvideo.attention.ring import ring_flash_attn_func
    from fastvideo.distributed import (
        cleanup_dist_env_and_memory,
        get_sp_group,
        maybe_init_distributed_environment_and_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_free_port()))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    try:
        maybe_init_distributed_environment_and_model_parallel(
            tp_size=1,
            sp_size=1,
            ring_size=1,
        )

        q, k, v = _make_qkv(
            batch_size=BATCH_SIZE,
            sequence_length=GLOBAL_SEQ_LEN,
            num_heads=NUM_HEADS,
            head_size=HEAD_SIZE,
            device=device,
        )

        softmax_scale = HEAD_SIZE**-0.5

        reference = flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=False,
        )

        # ring_size=1 disables the configured Ring group, so use the
        # one-rank SP process group directly to test kernel degeneration.
        ring_output = ring_flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=False,
            group=get_sp_group().device_group,
        )

        _assert_attention_close(
            ring_output,
            reference,
        )

    finally:
        cleanup_dist_env_and_memory()


def test_blockwise_lse_merge_matches_full_attention() -> None:
    """Blockwise attention merged through LSE must match full attention.

    This simulates the numerical part of Ring Attention on one GPU:

        Q attends to K0/V0
        Q attends to K1/V1
        partial outputs are merged using log-sum-exp

    The merged result must equal:

        Q attends to concat(K0, K1) / concat(V0, V1)

    This test validates the core online-softmax logic without requiring
    distributed communication.
    """

    if not torch.cuda.is_available():
        pytest.skip("This test requires CUDA.")

    from flash_attn import flash_attn_func

    from fastvideo.attention.ring._fa_kernels import _fa_forward
    from fastvideo.attention.ring.utils import update_out_and_lse

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    generator = torch.Generator(device="cpu").manual_seed(SEED)

    q = torch.randn(
        BATCH_SIZE,
        128,
        NUM_HEADS,
        HEAD_SIZE,
        generator=generator,
        dtype=torch.bfloat16,
    ).to(device)

    k0 = torch.randn(
        BATCH_SIZE,
        128,
        NUM_HEADS,
        HEAD_SIZE,
        generator=generator,
        dtype=torch.bfloat16,
    ).to(device)

    v0 = torch.randn(
        BATCH_SIZE,
        128,
        NUM_HEADS,
        HEAD_SIZE,
        generator=generator,
        dtype=torch.bfloat16,
    ).to(device)

    k1 = torch.randn(
        BATCH_SIZE,
        128,
        NUM_HEADS,
        HEAD_SIZE,
        generator=generator,
        dtype=torch.bfloat16,
    ).to(device)

    v1 = torch.randn(
        BATCH_SIZE,
        128,
        NUM_HEADS,
        HEAD_SIZE,
        generator=generator,
        dtype=torch.bfloat16,
    ).to(device)

    softmax_scale = HEAD_SIZE**-0.5

    block_out_0, block_lse_0 = _fa_forward(
        q,
        k0,
        v0,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=False,
    )

    block_out_1, block_lse_1 = _fa_forward(
        q,
        k1,
        v1,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=False,
    )

    merged_out, merged_lse = update_out_and_lse(
        None,
        None,
        block_out_0,
        block_lse_0,
    )

    merged_out, merged_lse = update_out_and_lse(
        merged_out,
        merged_lse,
        block_out_1,
        block_lse_1,
    )

    del merged_lse

    full_output = flash_attn_func(
        q,
        torch.cat([k0, k1], dim=1),
        torch.cat([v0, v1], dim=1),
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=False,
    )

    _assert_attention_close(merged_out, full_output)


def _run_multi_gpu_worker(output_path: Path) -> None:
    """Run the actual distributed Ring Attention parity test."""

    import torch.distributed as dist
    from flash_attn import flash_attn_func

    from fastvideo.attention.ring import ring_flash_attn_func
    from fastvideo.distributed import (
        cleanup_dist_env_and_memory,
        get_ring_group,
        maybe_init_distributed_environment_and_model_parallel,
    )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    try:
        maybe_init_distributed_environment_and_model_parallel(
            tp_size=1,
            sp_size=world_size,
            ring_size=world_size,
        )

        if GLOBAL_SEQ_LEN % world_size != 0:
            raise ValueError(
                f"GLOBAL_SEQ_LEN={GLOBAL_SEQ_LEN} must be divisible by "
                f"world_size={world_size}."
            )

        local_seq_len = GLOBAL_SEQ_LEN // world_size

        q_global, k_global, v_global = _make_qkv(
            batch_size=BATCH_SIZE,
            sequence_length=GLOBAL_SEQ_LEN,
            num_heads=NUM_HEADS,
            head_size=HEAD_SIZE,
            device=device,
        )

        # Ensure all ranks use identical global tensors.
        dist.broadcast(q_global, src=0)
        dist.broadcast(k_global, src=0)
        dist.broadcast(v_global, src=0)

        start = rank * local_seq_len
        end = start + local_seq_len

        q_local = q_global[:, start:end].contiguous()
        k_local = k_global[:, start:end].contiguous()
        v_local = v_global[:, start:end].contiguous()

        softmax_scale = HEAD_SIZE**-0.5

        ring_output_local = ring_flash_attn_func(
            q_local,
            k_local,
            v_local,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=False,
            group=get_ring_group(),
        )

        gathered_outputs = [
            torch.empty_like(ring_output_local)
            for _ in range(world_size)
        ]

        dist.all_gather(
            gathered_outputs,
            ring_output_local,
        )

        ring_output_global = torch.cat(
            gathered_outputs,
            dim=1,
        )

        if rank == 0:
            reference = flash_attn_func(
                q_global,
                k_global,
                v_global,
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=False,
            )

            torch.save(
                {
                    "ring_output": ring_output_global.cpu(),
                    "reference": reference.cpu(),
                },
                output_path,
            )

        dist.barrier()

    finally:
        cleanup_dist_env_and_memory()


def _run_torchrun(
    script_path: Path,
    nproc_per_node: int,
    output_path: Path,
) -> None:
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

    process = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    if process.returncode != 0:
        raise RuntimeError(
            f"Ring worker failed with code {process.returncode}\n"
            f"STDOUT:\n{process.stdout}\n"
            f"STDERR:\n{process.stderr}"
        )


def test_multi_gpu_ring_attention_matches_full_attention(
    tmp_path: Path,
) -> None:
    """Validate actual KV communication when two GPUs are available."""

    if not torch.cuda.is_available():
        pytest.skip("This test requires CUDA.")

    if torch.cuda.device_count() < MULTI_GPU_RING_WORLD_SIZE:
        pytest.skip(
            "Multi-GPU Ring Attention test requires at least "
            f"{MULTI_GPU_RING_WORLD_SIZE} CUDA devices."
        )

    script_path = Path(__file__).resolve()
    output_path = tmp_path / "ring_vs_full.pt"

    _run_torchrun(
        script_path=script_path,
        nproc_per_node=MULTI_GPU_RING_WORLD_SIZE,
        output_path=output_path,
    )

    saved = torch.load(
        output_path,
        map_location="cpu",
        weights_only=True,
    )

    ring_output = saved["ring_output"]
    reference = saved["reference"]

    _assert_attention_close(
        ring_output,
        reference,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ring-worker",
        action="store_true",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )
    return parser.parse_args()


# Single GPU:
# CUDA_VISIBLE_DEVICES=0 pytest -sv \
#   fastvideo/tests/distributed/test_ring_attention.py
#
# Multi GPU:
# CUDA_VISIBLE_DEVICES=0,1 pytest -sv \
#   fastvideo/tests/distributed/test_ring_attention.py
if __name__ == "__main__":
    args = _parse_args()

    if not args.ring_worker:
        raise SystemExit(
            "This module is intended to be run by pytest."
        )

    if args.output is None:
        raise SystemExit(
            "--output is required in worker mode."
        )

    _run_multi_gpu_worker(
        Path(args.output),
    )