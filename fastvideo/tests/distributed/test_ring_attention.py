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
import signal
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
    shape = (
        batch_size,
        sequence_length,
        num_heads,
        head_size,
    )

    q = torch.randn(
        shape,
        generator=torch.Generator(device="cpu").manual_seed(SEED),
        dtype=dtype,
    ).to(device)

    k = torch.randn(
        shape,
        generator=torch.Generator(device="cpu").manual_seed(SEED + 1),
        dtype=dtype,
    ).to(device)

    v = torch.randn(
        shape,
        generator=torch.Generator(device="cpu").manual_seed(SEED + 2),
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


def _log(rank: int, message: str) -> None:
    """Print an unbuffered worker log line for hang diagnosis."""
    print(f"[rank {rank}] {message}", flush=True)


def test_ring_attention_world_size_one_matches_flash_attention() -> None:
    if not torch.cuda.is_available():
        pytest.skip("This test requires CUDA.")

    import torch.distributed as dist
    from flash_attn import flash_attn_func

    from fastvideo.attention.ring import ring_flash_attn_func

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(_free_port())
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=0,
            world_size=1,
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

        ring_output = ring_flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=False,
            group=dist.group.WORLD,
        )

        _assert_attention_close(ring_output, reference)

    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


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
    import torch.distributed as dist
    from flash_attn import flash_attn_func

    from fastvideo.attention.ring import ring_flash_attn_func

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    def log(message: str) -> None:
        print(f"[rank {rank}] {message}", flush=True)

    try:
        log("initializing torch distributed process group")

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )

        log("torch distributed process group initialized")

        if GLOBAL_SEQ_LEN % world_size != 0:
            raise ValueError(
                f"GLOBAL_SEQ_LEN={GLOBAL_SEQ_LEN} must be divisible by "
                f"world_size={world_size}."
            )

        local_seq_len = GLOBAL_SEQ_LEN // world_size

        # Every rank generates identical Q/K/V independently.
        q_global, k_global, v_global = _make_qkv(
            batch_size=BATCH_SIZE,
            sequence_length=GLOBAL_SEQ_LEN,
            num_heads=NUM_HEADS,
            head_size=HEAD_SIZE,
            device=device,
        )

        torch.cuda.synchronize(device)
        log("deterministic global QKV created")

        # Temporary NCCL P2P sanity check.
        send_tensor = torch.tensor(
            [float(rank)],
            device=device,
            dtype=torch.float32,
        )
        recv_tensor = torch.empty_like(send_tensor)

        next_rank = (rank + 1) % world_size
        prev_rank = (rank - 1 + world_size) % world_size

        log(
            f"submitting raw P2P: send->{next_rank}, "
            f"recv<-{prev_rank}"
        )

        requests = dist.batch_isend_irecv(
            [
                dist.P2POp(
                    dist.isend,
                    send_tensor,
                    next_rank,
                ),
                dist.P2POp(
                    dist.irecv,
                    recv_tensor,
                    prev_rank,
                ),
            ]
        )

        for request in requests:
            request.wait()

        torch.cuda.synchronize(device)

        assert recv_tensor.item() == float(prev_rank)
        log("minimal raw P2P test passed")

        start = rank * local_seq_len
        end = start + local_seq_len

        q_local = q_global[:, start:end].contiguous()
        k_local = k_global[:, start:end].contiguous()
        v_local = v_global[:, start:end].contiguous()

        softmax_scale = HEAD_SIZE**-0.5

        log("entering ring_flash_attn_func")

        ring_output_local = ring_flash_attn_func(
            q_local,
            k_local,
            v_local,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=False,
            group=dist.group.WORLD,
        )

        torch.cuda.synchronize(device)
        log("ring_flash_attn_func complete")

        gathered_outputs = [
            torch.empty_like(ring_output_local)
            for _ in range(world_size)
        ]

        dist.all_gather(
            gathered_outputs,
            ring_output_local,
        )

        torch.cuda.synchronize(device)
        log("all_gather complete")

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

            torch.cuda.synchronize(device)

            torch.save(
                {
                    "ring_output": ring_output_global.cpu(),
                    "reference": reference.cpu(),
                },
                output_path,
            )

            log("reference saved")

    finally:
        if dist.is_available() and dist.is_initialized():
            log("destroying torch distributed process group")
            dist.destroy_process_group()


def _run_torchrun(
    script_path: Path,
    nproc_per_node: int,
    output_path: Path,
) -> None:
    cmd = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={nproc_per_node}",
        str(script_path),
        "--ring-worker",
        "--output",
        str(output_path),
    ]

    env = os.environ.copy()

    # The single-GPU test initializes an env:// process group in the pytest
    # process. Do not pass those one-rank values into the torchrun launcher.
    # torchrun will generate the correct rendezvous and rank variables.
    for name in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        env.pop(name, None)

    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "NCCL_DEBUG": env.get("NCCL_DEBUG", "INFO"),
            "TORCH_DISTRIBUTED_DEBUG": env.get(
                "TORCH_DISTRIBUTED_DEBUG",
                "DETAIL",
            ),
            "NCCL_ASYNC_ERROR_HANDLING": "1",
        }
    )

    # Some multi-GPU hosts advertise CUDA peer access between devices that
    # are not actually NVLink/PCIe-bridge connected (e.g. topology "SYS",
    # crossing NUMA nodes). Real P2P there can hang indefinitely. Default to
    # disabling NCCL P2P so the test is hang-safe on such hosts, but respect
    # an explicit caller override.
    env.setdefault("NCCL_P2P_DISABLE", "1")

    # torchrun spawns the per-rank worker processes as children that stay in
    # this process's group. Launch it as its own session leader so a timeout
    # can kill the whole tree via killpg -- subprocess.run(timeout=...) only
    # terminates the torchrun launcher itself, leaving orphaned rank workers
    # that keep the GPUs pegged at 100% util for every subsequent test run.
    proc = subprocess.Popen(
        cmd,
        env=env,
        start_new_session=True,
    )

    try:
        # Do not capture stdout/stderr: per-rank progress logs must remain
        # visible so a communication hang can be located immediately.
        returncode = proc.wait(timeout=120)
    except subprocess.TimeoutExpired as exc:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()
        raise RuntimeError(
            "The multi-GPU Ring Attention worker timed out after 120 seconds "
            "and its process group was killed. Use the last printed per-rank "
            "stage to determine whether the hang occurred during "
            "initialization, Ring P2P communication, all_gather, or the "
            "final barrier."
        ) from exc

    if returncode != 0:
        raise RuntimeError(
            f"Ring Attention worker exited with code {returncode}."
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