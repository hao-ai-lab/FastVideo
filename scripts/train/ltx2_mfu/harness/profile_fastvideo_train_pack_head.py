#!/usr/bin/env python3
"""Rank-zero Kineto trace with kernel-category rollup for the packed harness.

Same wrapping approach as ``profile_fastvideo_train_pack_fa47ce1.py`` but adds
a per-category CUDA-time decomposition (gemm / attention / optimizer / comm /
memcpy / compiled-fused / eager elementwise / reduce+norm / other) so the
non-GEMM, non-attention band can be attributed by name. Diagnostics only,
never MFU evidence.
"""

from __future__ import annotations

from collections import defaultdict
import builtins
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

import benchmark_fastvideo_train_pack_d016 as benchmark
import fastvideo.train.trainer as trainer_module

WAIT_STEPS = 10
WARMUP_STEPS = 1
ACTIVE_STEPS = 2
TOTAL_STEPS = 14
TOP_ROWS = 40
OUTPUT_PREFIX = Path(os.environ.get("FASTVIDEO_KINETO_PREFIX", "/mnt/pr1630_pack_head_rank0"))


def _self_cuda_us(event: Any) -> float:
    for name in ("self_device_time_total", "self_cuda_time_total"):
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def _category(name: str) -> str:
    lowered = name.lower()
    if "nccl" in lowered:
        return "comm"
    if "memcpy" in lowered or "memset" in lowered:
        return "memcpy"
    if any(key in lowered for key in ("nvjet", "cutlass", "gemm", "cublas", "matmul", "splitk")):
        return "gemm"
    if any(key in lowered for key in ("flash", "fmha", "cute", "attention", "_attn")):
        return "attention"
    if "adam" in lowered or "multi_tensor" in lowered:
        return "optimizer"
    if lowered.startswith("triton_"):
        return "compiled_fused"
    if "elementwise" in lowered or "vectorized" in lowered:
        return "eager_elementwise"
    if any(key in lowered for key in ("reduce", "norm", "welford", "softmax")):
        return "reduce_norm"
    return "other"


def _trace_ready(profiler: torch.profiler.profile) -> None:
    trace_path = OUTPUT_PREFIX.with_suffix(".trace.json.gz")
    summary_path = OUTPUT_PREFIX.with_suffix(".summary.json")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(trace_path))

    kernels: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for event in profiler.events():
        if not str(getattr(event, "device_type", "")).endswith("CUDA"):
            continue
        row = kernels[event.name]
        row[0] += 1
        row[1] += _self_cuda_us(event)
    if not kernels:
        raise RuntimeError("Kineto captured no CUDA kernel events")

    categories: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for name, (count, cuda_us) in kernels.items():
        row = categories[_category(name)]
        row[0] += count
        row[1] += cuda_us
    category_rows = [{
        "category": category,
        "calls_per_step": round(count / ACTIVE_STEPS, 1),
        "cuda_ms_per_step": round(cuda_us / 1000.0 / ACTIVE_STEPS, 3),
    } for category, (count, cuda_us) in sorted(categories.items(), key=lambda item: item[1][1], reverse=True)]

    kernel_rows = [{
        "name": name[:160],
        "calls_per_step": round(count / ACTIVE_STEPS, 1),
        "cuda_ms_per_step": round(cuda_us / 1000.0 / ACTIVE_STEPS, 3),
        "category": _category(name),
    } for name, (count, cuda_us) in sorted(kernels.items(), key=lambda item: item[1][1], reverse=True)[:TOP_ROWS]]

    summary = {
        "diagnostic_only_not_mfu_evidence": True,
        "rank": 0,
        "active_steps": ACTIVE_STEPS,
        "trace": str(trace_path),
        "trace_bytes": trace_path.stat().st_size,
        "total_cuda_ms_per_step": round(
            sum(row[1] for row in kernels.values()) / 1000.0 / ACTIVE_STEPS, 3),
        "category_rollup": category_rows,
        "top_cuda_kernels": kernel_rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    builtins.print("KINETO_CATEGORIES " + json.dumps(summary, separators=(",", ":")), flush=True)


class _ProfilerStep:

    def __init__(self, profiler: torch.profiler.profile) -> None:
        self.profiler = profiler

    def on_training_step_end(self, _method: Any, _metrics: dict[str, Any], iteration: int = 0) -> None:
        del iteration
        self.profiler.step()


def _quiet_benchmark_print(*values: Any, **kwargs: Any) -> None:
    if values and str(values[0]).startswith("BF16_"):
        return
    builtins.print(*values, **kwargs)


def main() -> None:
    benchmark.EXPECTED_STEPS = TOTAL_STEPS
    benchmark.print = _quiet_benchmark_print
    original_run = trainer_module.Trainer.run

    def _profiled_run(self: Any, method: Any, **kwargs: Any) -> Any:
        if not dist.is_initialized() or dist.get_rank() != 0:
            return original_run(self, method, **kwargs)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=WAIT_STEPS, warmup=WARMUP_STEPS, active=ACTIVE_STEPS, repeat=1),
            on_trace_ready=_trace_ready,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
        ) as profiler:
            self.callbacks._callbacks["_kineto_step"] = _ProfilerStep(profiler)
            result = original_run(self, method, **kwargs)
            torch.cuda.synchronize()
            return result

    trainer_module.Trainer.run = _profiled_run
    benchmark.main()


if __name__ == "__main__":
    main()
