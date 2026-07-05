# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
from torch.distributed import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard
from torch.distributed.tensor import DTensor

from fastvideo.layers.layernorm import RMSNorm

WORLD_SIZE = 2
HIDDEN_SIZE = 8
SEED = 1379
REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_torchrun(script_path: Path, mode: str, output_path: Path) -> None:
    # --standalone binds the rendezvous port atomically, avoiding the
    # free-port-probe race a hand-picked --master_port would have.
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        str(WORLD_SIZE),
        str(script_path),
        "--rmsnorm-fsdp-worker",
        "--mode",
        mode,
        "--output",
        str(output_path),
    ]
    env = os.environ.copy()
    env.setdefault("TORCHDYNAMO_DISABLE", "1")
    try:
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"{mode} worker timed out after 120 seconds\n"
            f"STDOUT:\n{error.stdout}\n"
            f"STDERR:\n{error.stderr}"
        ) from error
    if process.returncode != 0:
        raise RuntimeError(
            f"{mode} worker failed with code {process.returncode}\n"
            f"STDOUT:\n{process.stdout}\n"
            f"STDERR:\n{process.stderr}"
        )


def _summarize_tensor(tensor: torch.Tensor | Any) -> dict[str, Any]:
    return {
        "type": type(tensor).__name__,
        "is_dtensor": isinstance(tensor, DTensor),
        "shape": list(tensor.shape) if hasattr(tensor, "shape") else None,
        "device": str(tensor.device) if hasattr(tensor, "device") else None,
        "dtype": str(tensor.dtype) if hasattr(tensor, "dtype") else None,
    }


def _run_worker(mode: str, output_path: Path) -> None:
    if mode not in {
        "module_no_offload",
        "direct_no_offload",
        "module_cpu_offload",
        "direct_cpu_offload",
    }:
        raise ValueError(f"Unsupported mode: {mode}")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(SEED + rank)

    try:
        mesh = init_device_mesh("cuda", (world_size,))
        norm = RMSNorm(HIDDEN_SIZE, eps=1e-6, has_weight=True).to(device)
        with torch.no_grad():
            norm.weight.fill_(1.0)

        fsdp_kwargs: dict[str, Any] = {"mesh": mesh}
        if mode.endswith("cpu_offload"):
            fsdp_kwargs["offload_policy"] = CPUOffloadPolicy(pin_memory=False)
        # fully_shard is applied to the bare RMSNorm to make the hook bypass
        # observable. Production sharding (fsdp_load.shard_model) only wraps
        # whole transformer blocks, whose pre-forward all-gather localizes norm
        # weights before the qk-norm call sites run, so this pins the dispatch
        # invariant rather than reproducing a production topology.
        fully_shard(norm, **fsdp_kwargs)

        x = torch.randn(2, 3, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
        call_kind = "direct" if mode.startswith("direct") else "module"

        try:
            if call_kind == "direct":
                output = norm.forward_native(x)
            else:
                output = norm(x)
            torch.cuda.synchronize(device)
            result = {
                "rank": rank,
                "ok": True,
                "mode": mode,
                "weight": _summarize_tensor(norm.weight),
                "output": _summarize_tensor(output),
            }
        except Exception as exc:
            result = {
                "rank": rank,
                "ok": False,
                "mode": mode,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "weight": _summarize_tensor(norm.weight),
            }

        gathered = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(result, object_gather_list=gathered, dst=0)
        if rank == 0:
            output_path.write_text(json.dumps(gathered, indent=2), encoding="utf-8")
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("mode", "expect_ok"),
    [
        ("module_no_offload", True),
        ("direct_no_offload", False),
        ("module_cpu_offload", True),
        ("direct_cpu_offload", False),
    ],
)
def test_rmsnorm_forward_native_bypasses_fsdp_hooks(mode: str, expect_ok: bool, tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("This test requires CUDA.")
    if torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip(f"This test requires at least {WORLD_SIZE} CUDA devices.")

    output_path = tmp_path / f"{mode}.json"
    _run_torchrun(Path(__file__).resolve(), mode, output_path)
    results = json.loads(output_path.read_text(encoding="utf-8"))
    print(f"\n{mode} results:\n{json.dumps(results, indent=2)}")

    if expect_ok:
        failures = [result for result in results if not result["ok"]]
        assert not failures, json.dumps(results, indent=2)
        return

    successes = [result for result in results if result["ok"]]
    assert not successes, json.dumps(results, indent=2)
    error_text = "\n".join(result.get("error", "") for result in results)
    # Pin the specific bypassed-hook failure: "got mixed torch.Tensor and
    # DTensor" ("Tensor" alone is a substring of "DTensor", so it adds nothing).
    assert "mixed" in error_text and "DTensor" in error_text, json.dumps(results, indent=2)


def test_no_direct_forward_native_calls_in_models() -> None:
    """Direct .forward_native(...) calls bypass nn.Module.__call__ and FSDP
    hooks (issue #1379); model code must use module dispatch instead."""
    models_dir = REPO_ROOT / "fastvideo" / "models"
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in sorted(models_dir.rglob("*.py"))
        if ".forward_native(" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"Replace .forward_native(...) with module dispatch in: {offenders}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rmsnorm-fsdp-worker", action="store_true")
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not args.rmsnorm_fsdp_worker:
        raise SystemExit("This module is intended to be run by pytest.")
    if args.mode is None or args.output is None:
        raise SystemExit("--mode and --output are required in worker mode.")
    _run_worker(mode=args.mode, output_path=Path(args.output))
