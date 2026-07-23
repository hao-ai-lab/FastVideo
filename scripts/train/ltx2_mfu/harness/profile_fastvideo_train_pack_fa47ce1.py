#!/usr/bin/env python3
"""Rank-zero Kineto trace for the packed LTX-2 training benchmark.

This wraps the already-gated benchmark harness so its semantic and optimizer
checks stay identical. Profiler timings are diagnostics, never MFU evidence.
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
TOP_ROWS = 20
OUTPUT_PREFIX = Path(os.environ.get("FASTVIDEO_KINETO_PREFIX", "/mnt/pr1630_pack_fa47ce1_rank0"))


def _cuda_time_us(event: Any, *, self_time: bool) -> float:
    prefixes = ("self_", "") if self_time else ("", "self_")
    for prefix in prefixes:
        for suffix in ("device_time_total", "cuda_time_total"):
            value = getattr(event, prefix + suffix, None)
            if value is not None:
                return float(value)
    return 0.0


def _top_cuda_rows(profiler: torch.profiler.profile) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    operators = sorted(
        profiler.key_averages(),
        key=lambda event: _cuda_time_us(event, self_time=True),
        reverse=True,
    )[:TOP_ROWS]
    operator_rows = [{
        "name": event.key,
        "calls": int(event.count),
        "self_cuda_ms": round(_cuda_time_us(event, self_time=True) / 1000.0, 3),
        "total_cuda_ms": round(_cuda_time_us(event, self_time=False) / 1000.0, 3),
    } for event in operators if _cuda_time_us(event, self_time=True) > 0]

    kernels: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for event in profiler.events():
        if not str(getattr(event, "device_type", "")).endswith("CUDA"):
            continue
        row = kernels[event.name]
        row[0] += 1
        row[1] += _cuda_time_us(event, self_time=True)
    kernel_rows = [{
        "name": name,
        "calls": int(count),
        "cuda_ms": round(cuda_us / 1000.0, 3),
    } for name, (count, cuda_us) in sorted(kernels.items(), key=lambda item: item[1][1], reverse=True)[:TOP_ROWS]]
    return operator_rows, kernel_rows


def _trace_ready(profiler: torch.profiler.profile) -> None:
    trace_path = OUTPUT_PREFIX.with_suffix(".trace.json.gz")
    summary_path = OUTPUT_PREFIX.with_suffix(".summary.json")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(trace_path))
    operators, kernels = _top_cuda_rows(profiler)
    if not operators or not kernels:
        raise RuntimeError("Kineto captured no CUDA operator or kernel events")
    summary = {
        "diagnostic_only_not_mfu_evidence": True,
        "rank": 0,
        "schedule": {
            "wait_steps": WAIT_STEPS,
            "warmup_steps": WARMUP_STEPS,
            "active_steps": ACTIVE_STEPS,
        },
        "trace": str(trace_path),
        "trace_bytes": trace_path.stat().st_size,
        "top_cuda_operators": operators,
        "top_cuda_kernels": kernels,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    builtins.print("KINETO_SUMMARY " + json.dumps(summary, separators=(",", ":")), flush=True)


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
