#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare complete FastVideo inference with FlashAttention and FlashInfer.

Each backend runs in a fresh subprocess because attention selection is resolved
when model components are constructed and CUDA state/model memory must not leak
between benchmark arms.

Example:
    CUDA_VISIBLE_DEVICES=0 python examples/inference/benchmark_attention_backends.py \
        --config scripts/inference/inference_wan.yaml \
        --override request.prompt="A fox running through snow" \
        --override request.inputs.prompt_path=null \
        --warmups 1 --repeats 3 --save-outputs
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

_BACKENDS = ("FLASH_ATTN", "FLASHINFER")
_RESULT_PREFIX = "FASTVIDEO_BACKEND_BENCHMARK_RESULT="


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Nested FastVideo inference YAML/JSON config")
    parser.add_argument("--override",
                        action="append",
                        default=[],
                        help="Dotted generator/request override; repeat as needed")
    parser.add_argument("--output-dir", default="outputs/attention_backend_comparison")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--save-outputs",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Save measured videos for visual comparison (included in wall time)")
    parser.add_argument("--backend-order", nargs=2, choices=_BACKENDS, default=list(_BACKENDS))
    parser.add_argument("--worker-backend", choices=_BACKENDS, help=argparse.SUPPRESS)
    return parser.parse_args()


def _synchronize_cuda(torch_module: Any) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def _run_worker(args: argparse.Namespace) -> None:
    # This must happen before importing FastVideo: backend selection is folded
    # into component construction and must stay fixed for the worker lifetime.
    backend = args.worker_backend
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = backend

    import torch

    from fastvideo import VideoGenerator
    from fastvideo.entrypoints.cli.inference_config import build_generate_run_config

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires an NVIDIA CUDA GPU")
    if backend == "FLASHINFER" and torch.cuda.get_device_capability() < (8, 0):
        raise RuntimeError("FLASHINFER requires an NVIDIA GPU with compute capability sm80 or newer")

    config_args = argparse.Namespace(config=args.config)
    config_overrides = [item if item.startswith("--") else f"--{item}" for item in args.override]
    run_config = build_generate_run_config(config_args, overrides=config_overrides)
    if run_config.generator.engine.num_gpus != 1:
        raise ValueError("This comparison script currently requires generator.engine.num_gpus=1")

    backend_dir = Path(args.output_dir).resolve() / backend.lower()
    backend_dir.mkdir(parents=True, exist_ok=True)

    load_started = time.perf_counter()
    generator = VideoGenerator.from_config(run_config.generator)
    load_seconds = time.perf_counter() - load_started

    def run_once(phase: str, index: int, *, measured: bool) -> float:
        request = deepcopy(run_config.request)
        request.output.output_path = str(backend_dir)
        request.output.output_video_name = f"{backend.lower()}_{phase}_{index:02d}"
        request.output.save_video = args.save_outputs if measured else False
        request.output.return_frames = False

        _synchronize_cuda(torch)
        started = time.perf_counter()
        generator.generate(request)
        _synchronize_cuda(torch)
        return time.perf_counter() - started

    warmup_seconds = [run_once("warmup", index, measured=False) for index in range(args.warmups)]
    measured_seconds = [run_once("run", index, measured=True) for index in range(args.repeats)]
    result = {
        "backend": backend,
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "device_capability": list(torch.cuda.get_device_capability()),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "config": str(Path(args.config).resolve()),
        "generator": asdict(run_config.generator),
        "request": asdict(run_config.request),
        "load_seconds": load_seconds,
        "warmup_seconds": warmup_seconds,
        "measured_seconds": measured_seconds,
        "median_seconds": statistics.median(measured_seconds),
        "mean_seconds": statistics.mean(measured_seconds),
        "save_outputs": args.save_outputs,
        "output_dir": str(backend_dir),
    }
    print(f"{_RESULT_PREFIX}{json.dumps(result, default=str)}", flush=True)


def _run_backend_subprocess(args: argparse.Namespace, backend: str) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        args.config,
        "--output-dir",
        args.output_dir,
        "--warmups",
        str(args.warmups),
        "--repeats",
        str(args.repeats),
        "--worker-backend",
        backend,
        "--save-outputs" if args.save_outputs else "--no-save-outputs",
    ]
    for override in args.override:
        command.extend(("--override", override))

    print(f"\n===== {backend} =====", flush=True)
    process = subprocess.Popen(command,
                               cwd=Path.cwd(),
                               env=os.environ.copy(),
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               text=True,
                               bufsize=1)
    result: dict[str, Any] | None = None
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        if line.startswith(_RESULT_PREFIX):
            result = json.loads(line[len(_RESULT_PREFIX):])
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{backend} worker failed with exit code {return_code}")
    if result is None:
        raise RuntimeError(f"{backend} worker exited without a benchmark result")
    return result


def _write_summary(args: argparse.Namespace, results: list[dict[str, Any]]) -> Path:
    by_backend = {result["backend"]: result for result in results}
    flash_seconds = by_backend["FLASH_ATTN"]["median_seconds"]
    flashinfer_seconds = by_backend["FLASHINFER"]["median_seconds"]
    summary = {
        "results": by_backend,
        "comparison": {
            "flash_attn_median_seconds": flash_seconds,
            "flashinfer_median_seconds": flashinfer_seconds,
            "flashinfer_speedup": flash_seconds / flashinfer_seconds,
            "flashinfer_time_change_percent": (flashinfer_seconds / flash_seconds - 1.0) * 100.0,
        },
        "timing_scope": "VideoGenerator.generate wall time with CUDA synchronization",
        "notes": [
            "Warmup runs are excluded from statistics.",
            "Saved-video encoding is included when --save-outputs is enabled.",
            "A speedup greater than 1.0 means FLASHINFER was faster.",
        ],
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    args = _parse_args()
    if args.warmups < 1:
        raise ValueError("--warmups must be at least 1 so JIT/caches are excluded")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if set(args.backend_order) != set(_BACKENDS):
        raise ValueError("--backend-order must contain FLASH_ATTN and FLASHINFER exactly once")
    if args.worker_backend is not None:
        _run_worker(args)
        return

    results = [_run_backend_subprocess(args, backend) for backend in args.backend_order]
    summary_path = _write_summary(args, results)
    comparison = json.loads(summary_path.read_text(encoding="utf-8"))["comparison"]
    print("\n===== Comparison =====")
    print(f"FLASH_ATTN median: {comparison['flash_attn_median_seconds']:.3f} s")
    print(f"FLASHINFER median:  {comparison['flashinfer_median_seconds']:.3f} s")
    print(f"FLASHINFER speedup: {comparison['flashinfer_speedup']:.3f}x")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
