# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the vendored Ring Attention implementation.
# -----------------------------------------------------------------------------
# Running the multi-GPU Ring Attention test
#
# Typical usage:
#
#   pytest -sv fastvideo/tests/distributed/test_ring_attention.py
#
# If the current environment has known NCCL P2P / IPC issues (e.g., some Docker
# containers), the test can be launched with NCCL socket fallback:
#
#   CUDA_VISIBLE_DEVICES=0,1 \
#   NCCL_CUMEM_ENABLE=0 \
#   NCCL_CUMEM_HOST_ENABLE=0 \
#   NCCL_P2P_DISABLE=1 \
#   NCCL_SHM_DISABLE=1 \
#   NCCL_SOCKET_IFNAME=eth0 \
#   pytest -sv \
#       fastvideo/tests/distributed/test_ring_attention.py::\
#test_multi_gpu_ring_attention_matches_full_attention
#
# These environment variables are only intended as an environment-specific
# workaround and should not be required on systems with a fully functional
# NCCL P2P setup.
# -----------------------------------------------------------------------------
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

from fastvideo import envs

SEED = 2026
MULTI_GPU_RING_WORLD_SIZE = 2

# Hybrid Ring x Ulysses (USP): ring_size=2, so ulysses_size = world_size //
# ring_size = 2. NUM_HEADS must be divisible by ulysses_size.
HYBRID_USP_WORLD_SIZE = 4
HYBRID_RING_SIZE = 2

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


def test_ring_attention_world_size_one_matches_flash_attention(
    env_overrides,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("This test requires CUDA.")

    import torch.distributed as dist
    from flash_attn import flash_attn_func

    from fastvideo.attention.ring import ring_flash_attn_func

    env_overrides.enter_context(envs.override_external("MASTER_ADDR", "127.0.0.1"))
    env_overrides.enter_context(envs.override_external("MASTER_PORT", str(_free_port())))
    env_overrides.enter_context(envs.override_external("RANK", "0"))
    env_overrides.enter_context(envs.override_external("LOCAL_RANK", "0"))
    env_overrides.enter_context(envs.override_external("WORLD_SIZE", "1"))

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
            softmax_scale=softmax_scale,
            causal=False,
        )

        ring_output = ring_flash_attn_func(
            q,
            k,
            v,
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

    from fastvideo.attention.ring.kernels.attention import (
        flash_attn_forward,
    )
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

    block_out_0, block_lse_0 = flash_attn_forward(
        q,
        k0,
        v0,
        softmax_scale=softmax_scale,
        causal=False,
        softcap=0.0,
    )

    block_out_1, block_lse_1 = flash_attn_forward(
        q,
        k1,
        v1,
        softmax_scale=softmax_scale,
        causal=False,
        softcap=0.0,
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
        softmax_scale=softmax_scale,
        causal=False,
    )

    _assert_attention_close(
        merged_out,
        full_output,
    )


def _run_multi_gpu_worker(output_path: Path) -> None:
    _run_production_worker(output_path, hybrid=False)


def _run_hybrid_usp_worker(output_path: Path) -> None:
    _run_production_worker(output_path, hybrid=True)


@torch.inference_mode()
def _run_production_worker(output_path: Path, *, hybrid: bool) -> None:
    """Exercise production dispatch, RoPE, padding and real Ring/USP communication."""
    import torch.distributed as dist
    import torch.nn.functional as F

    from fastvideo.attention.layer import DistributedAttention
    from fastvideo.attention.selector import _component_attention_backend_scope
    from flash_attn import flash_attn_func
    from fastvideo.distributed import cleanup_dist_env_and_memory
    from fastvideo.distributed.parallel_state import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.layers.rotary_embedding import _apply_rotary_emb
    from fastvideo.platforms import AttentionBackendEnum

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    cases = []
    try:
        maybe_init_distributed_environment_and_model_parallel(
            tp_size=1, sp_size=world_size,
            ring_size=HYBRID_RING_SIZE if hybrid else world_size,
        )
        with _component_attention_backend_scope(AttentionBackendEnum.FLASH_ATTN):
            attention = DistributedAttention(
                num_heads=NUM_HEADS, head_size=HEAD_SIZE,
                supported_attention_backends=(AttentionBackendEnum.FLASH_ATTN,),
            ).eval()
        assert attention.use_ring_attention

        # The last case leaves whole Ring chunks empty. It catches accidental
        # empty-KV kernel calls and NaNs from merging two -inf LSE blocks.
        for seq_len in (GLOBAL_SEQ_LEN, GLOBAL_SEQ_LEN - 1, 1):
            local_seq_len = (seq_len + world_size - 1) // world_size
            padded_len = local_seq_len * world_size
            q, k, v = _make_qkv(
                batch_size=BATCH_SIZE, sequence_length=seq_len,
                num_heads=NUM_HEADS, head_size=HEAD_SIZE, device=device,
            )
            angles = torch.randn(seq_len, HEAD_SIZE,
                                 generator=torch.Generator().manual_seed(SEED + 3)).to(device)
            freqs = (angles.cos(), angles.sin())
            reference = flash_attn_func(
                _apply_rotary_emb(q, *freqs, is_neox_style=False),
                _apply_rotary_emb(k, *freqs, is_neox_style=False), v,
                softmax_scale=HEAD_SIZE**-0.5, causal=False,
            )
            start = rank * local_seq_len
            shards = [F.pad(t, (0, 0, 0, 0, 0, padded_len - seq_len))
                      [:, start:start + local_seq_len].contiguous() for t in (q, k, v)]
            output, replicated_output = attention(
                *shards, original_seq_len=seq_len, freqs_cis=freqs,
            )
            assert replicated_output is None
            assert output.shape == shards[0].shape
            gathered = [torch.empty_like(output) for _ in range(world_size)]
            dist.all_gather(gathered, output.contiguous())
            actual = torch.cat(gathered, dim=1)
            _assert_attention_close(actual[:, :seq_len], reference)
            assert torch.count_nonzero(actual[:, seq_len:]) == 0
            if rank == 0:
                cases.append({"output": actual[:, :seq_len].cpu(), "reference": reference.cpu()})
            _log(rank, f"production parity passed: hybrid={hybrid}, seq_len={seq_len}")
        if rank == 0:
            torch.save(cases, output_path)
        dist.barrier()
        torch.cuda.synchronize(device)
    finally:
        cleanup_dist_env_and_memory()


def _run_torchrun(
    script_path: Path,
    nproc_per_node: int,
    output_path: Path,
    worker_flag: str = "--ring-worker",
) -> None:
    cmd = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={nproc_per_node}",
        str(script_path),
        worker_flag,
        "--output",
        str(output_path),
    ]

    # Inherit the caller's environment without copying it into Python. GNU env
    # removes stale rendezvous values only in the child; torchrun supplies new ones.
    launcher = ["env"]
    for name in (
        "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
        "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
    ):
        launcher.extend(["-u", name])
    launcher.extend(["PYTHONUNBUFFERED=1", "TORCH_NCCL_ASYNC_ERROR_HANDLING=1"])

    # Preserve explicitly supplied debug settings. DETAIL can deadlock NCCL
    # P2P on some systems, so verbose defaults remain opt-in.
    if envs.FASTVIDEO_TEST_RING_DEBUG.get():
        if os.environ.get("NCCL_DEBUG") is None:
            launcher.append("NCCL_DEBUG=INFO")
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            launcher.append("TORCH_DISTRIBUTED_DEBUG=DETAIL")
    cmd = launcher + cmd

    # torchrun spawns the per-rank worker processes as children that stay in
    # this process's group. Launch it as its own session leader so a timeout
    # can kill the whole tree via killpg -- subprocess.run(timeout=...) only
    # terminates the torchrun launcher itself, leaving orphaned rank workers
    # that keep the GPUs pegged at 100% util for every subsequent test run.
    proc = subprocess.Popen(
        cmd,
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


@pytest.mark.parametrize(
    ("debug_enabled", "expected_nccl", "expected_torch"),
    [
        (False, None, None),
        (True, "INFO", "DETAIL"),
    ],
)
@pytest.mark.parametrize("caller_debug", [None, "WARN"])
def test_torchrun_debug_defaults_are_opt_in(
    env_overrides,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    debug_enabled: bool,
    caller_debug: str | None,
    expected_nccl: str | None,
    expected_torch: str | None,
) -> None:
    """The harness must not enable distributed debug wrappers by default."""
    captured_cmd: list[str] = []

    class _CompletedProcess:
        pid = 1

        @staticmethod
        def wait(timeout: int | None = None) -> int:
            del timeout
            return 0

    def fake_popen(
        cmd: list[str],
        *,
        start_new_session: bool,
    ) -> _CompletedProcess:
        del start_new_session
        captured_cmd.extend(cmd)
        return _CompletedProcess()

    env_overrides.enter_context(envs.override_external("NCCL_DEBUG", caller_debug))
    env_overrides.enter_context(envs.override_external("TORCH_DISTRIBUTED_DEBUG", "OFF" if caller_debug else None))
    env_overrides.enter_context(envs.FASTVIDEO_TEST_RING_DEBUG.override(debug_enabled))
    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    _run_torchrun(
        script_path=Path(__file__),
        nproc_per_node=2,
        output_path=tmp_path / "unused.pt",
    )

    overrides = dict(arg.split("=", 1) for arg in captured_cmd[1:captured_cmd.index("torchrun")] if "=" in arg)
    assert overrides.get("NCCL_DEBUG") == (None if caller_debug else expected_nccl)
    assert overrides.get("TORCH_DISTRIBUTED_DEBUG") == (None if caller_debug else expected_torch)
    assert os.environ.get("NCCL_DEBUG") == caller_debug
    assert os.environ.get("TORCH_DISTRIBUTED_DEBUG") == ("OFF" if caller_debug else None)
    assert captured_cmd[0] == "env"
    assert captured_cmd[1:5] == ["-u", "RANK", "-u", "LOCAL_RANK"]


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

    for case in saved:
        _assert_attention_close(case["output"], case["reference"])


def test_multi_gpu_hybrid_usp_matches_full_attention(
    tmp_path: Path,
) -> None:
    """Validate the Ring x Ulysses (USP) hybrid (1 < ring_size < sp_size).

    ``_check_ring_attention_args`` accepts this configuration and
    ``--ring-size`` documents it as supported, but
    ``test_multi_gpu_ring_attention_matches_full_attention`` above only
    exercises pure Ring (``ring_size == sp_size``) and
    ``test_usp_group_layout.py`` only checks the rank-mesh arithmetic as a
    pure function. This is the numerical parity check for the hybrid path
    itself, driving real Ulysses all-to-all + Ring KV communication across
    four GPUs.
    """

    if not torch.cuda.is_available():
        pytest.skip("This test requires CUDA.")

    if torch.cuda.device_count() < HYBRID_USP_WORLD_SIZE:
        pytest.skip(
            "Hybrid USP test requires at least "
            f"{HYBRID_USP_WORLD_SIZE} CUDA devices."
        )

    script_path = Path(__file__).resolve()
    output_path = tmp_path / "usp_vs_full.pt"

    _run_torchrun(
        script_path=script_path,
        nproc_per_node=HYBRID_USP_WORLD_SIZE,
        output_path=output_path,
        worker_flag="--hybrid-usp-worker",
    )

    saved = torch.load(
        output_path,
        map_location="cpu",
        weights_only=True,
    )

    for case in saved:
        _assert_attention_close(case["output"], case["reference"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ring-worker",
        action="store_true",
    )
    parser.add_argument(
        "--hybrid-usp-worker",
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
# Multi GPU (normal NCCL path):
# CUDA_VISIBLE_DEVICES=0,1 pytest -sv \
#   fastvideo/tests/distributed/test_ring_attention.py
#
# Multi GPU (correctness-only socket fallback):
# FASTVIDEO_RING_TEST_SOCKET_FALLBACK=1 CUDA_VISIBLE_DEVICES=0,1 pytest -sv \
#   fastvideo/tests/distributed/test_ring_attention.py
if __name__ == "__main__":
    args = _parse_args()

    if not args.ring_worker and not args.hybrid_usp_worker:
        raise SystemExit(
            "This module is intended to be run by pytest."
        )

    if args.output is None:
        raise SystemExit(
            "--output is required in worker mode."
        )

    if args.hybrid_usp_worker:
        _run_hybrid_usp_worker(
            Path(args.output),
        )
    else:
        _run_multi_gpu_worker(
            Path(args.output),
        )
