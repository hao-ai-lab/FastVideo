#!/usr/bin/env python3
"""One-GPU exact-shape BF16 GEMM gate for packed LTX-2 training.

Run each backend in a fresh process.  The first invocation of every compiled
shape/phase is reported as cold compile time and is never included in timing.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


BLOCKS = 48
DTYPE = torch.bfloat16


@dataclass(frozen=True)
class Shape:
    key: str
    m: int
    k: int
    n: int
    roles: tuple[str, ...]


# The seven packed projections reduce to five unique shapes.  Each role occurs
# once per transformer block, once in fprop, dgrad, and wgrad.
SHAPES = (
    Shape("self_qkv", 4290, 4096, 12288, ("self_qkv", )),
    Shape("video_4096", 4290, 4096, 4096, ("self_out", "cross_q", "cross_out")),
    Shape("text_kv", 1024, 4096, 8192, ("text_kv", )),
    Shape("ffn_up", 4290, 4096, 16384, ("ffn_up", )),
    Shape("ffn_down", 4290, 16384, 4096, ("ffn_down", )),
)
PHASES = ("fprop", "dgrad", "wgrad")
EXPECTED_TFLOP_PER_STEP = 300.095807422464
EXPECTED_LOGICAL_GEMMS_PER_STEP = 7 * BLOCKS * len(PHASES)


def _fprop(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return F.linear(x, w, bias)


def _dgrad(dy: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.mm(dy, w)


def _wgrad(dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.mm(dy.t(), x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("current", "cutlass"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--inner", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1630)
    parser.add_argument("--batch-factor", type=int, default=1)
    parser.add_argument(
        "--compare",
        nargs=3,
        type=Path,
        metavar=("CURRENT_A", "CUTLASS", "CURRENT_B"),
        help="compare three completed JSONL files instead of running CUDA",
    )
    args = parser.parse_args()
    if (args.variant is None) == (args.compare is None):
        parser.error("specify exactly one of --variant or --compare")
    if min(args.warmup, args.samples, args.inner, args.batch_factor) < 1:
        parser.error("--warmup, --samples, --inner, and --batch-factor must be positive")
    return args


class Jsonl:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", encoding="utf-8")

    def emit(self, kind: str, **values: Any) -> None:
        line = json.dumps({"kind": kind, **values}, sort_keys=True)
        print(line, flush=True)
        self._file.write(line + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def git_head(path: str | None) -> str | None:
    if not path:
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def config_proof(variant: str) -> dict[str, Any]:
    from torch._inductor import config
    from torch._inductor.codegen.cutlass.utils import try_import_cutlass

    swizzles_env = os.environ.get("CUTLASS_SWIZZLES")
    if swizzles_env:
        config.cutlass.cutlass_max_profiling_swizzle_options = [int(value) for value in swizzles_env.split(",")]
    cutlass_available = try_import_cutlass()
    proof = {
        "variant": variant,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "fastvideo_commit": os.environ.get("FASTVIDEO_COMMIT"),
        "max_autotune_gemm": config.max_autotune_gemm,
        "max_autotune_gemm_backends": config.max_autotune_gemm_backends,
        "max_autotune_gemm_search_space": config.max_autotune_gemm_search_space,
        "inductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        "cutlass_dir": config.cutlass.cutlass_dir,
        "cutlass_commit": git_head(config.cutlass.cutlass_dir),
        "cutlass_importable": cutlass_available,
        "cutlass_enabled_ops": config.cutlass.cutlass_enabled_ops,
        "cutlass_instantiation_level": config.cutlass.cutlass_instantiation_level,
        "cutlass_allowlist": config.cutlass.cutlass_op_allowlist_regex,
        "cutlass_denylist": config.cutlass.cutlass_op_denylist_regex,
        "cutlass_swizzles": list(config.cutlass.cutlass_max_profiling_swizzle_options),
        "cublas_preferred_backend": str(torch.backends.cuda.preferred_blas_library()),
        "env": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM",
                "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS",
                "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE",
                "TORCHINDUCTOR_CUTLASS_DIR",
                "TORCHINDUCTOR_CUTLASS_INSTANTIATION_LEVEL",
                "CUTLASS_EPILOGUE_FUSION",
                "CUTLASS_SWIZZLES",
            )
        },
    }
    if variant == "cutlass":
        if not config.max_autotune_gemm:
            raise RuntimeError("cutlass variant requires TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1")
        backends = {item.strip() for item in config.max_autotune_gemm_backends.split(",")}
        if backends != {"ATEN", "CUTLASS"}:
            raise RuntimeError(f"expected candidate backends ATEN,CUTLASS; got {sorted(backends)}")
        if not cutlass_available:
            raise RuntimeError(f"PyTorch cannot import CUTLASS from {config.cutlass.cutlass_dir}")
        if list(config.cutlass.cutlass_max_profiling_swizzle_options) != [4]:
            raise RuntimeError("this safety gate requires exactly CUTLASS_SWIZZLES=4")
    elif config.max_autotune_gemm:
        raise RuntimeError("current variant requires TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=0")
    return proof


def make_inputs(shape: Shape, seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x = torch.empty((shape.m, shape.k), device="cuda", dtype=DTYPE).normal_()
    w = torch.empty((shape.n, shape.k), device="cuda", dtype=DTYPE).normal_()
    w.mul_(1.0 / math.sqrt(shape.k))
    bias = torch.empty((shape.n, ), device="cuda", dtype=DTYPE).normal_(std=0.01)
    dy = torch.empty((shape.m, shape.n), device="cuda", dtype=DTYPE).normal_()
    return {"x": x, "w": w, "bias": bias, "dy": dy}


def phase_call(
    phase: str,
    tensors: dict[str, torch.Tensor],
) -> tuple[Callable[..., torch.Tensor], tuple[torch.Tensor, ...]]:
    if phase == "fprop":
        return _fprop, (tensors["x"], tensors["w"], tensors["bias"])
    if phase == "dgrad":
        return _dgrad, (tensors["dy"], tensors["w"])
    if phase == "wgrad":
        return _wgrad, (tensors["dy"], tensors["x"])
    raise AssertionError(phase)


@torch.no_grad()
def parity(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    candidate_f = candidate.float()
    reference_f = reference.float()
    delta = candidate_f - reference_f
    max_abs = delta.abs().max().item()
    max_ref = reference_f.abs().max().item()
    relative_l2 = (delta.norm() / reference_f.norm().clamp_min(1.0e-12)).item()
    passed = relative_l2 <= 0.02 and max_abs <= max(1.0, 0.03 * max_ref)
    return {
        "passed": passed,
        "max_abs": max_abs,
        "max_ref": max_ref,
        "relative_l2": relative_l2,
    }


@torch.no_grad()
def time_cuda(
    fn: Callable[..., torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
    warmup: int,
    samples: int,
    inner: int,
) -> list[float]:
    for _ in range(warmup):
        fn(*inputs)
    torch.cuda.synchronize()
    timings = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            output = fn(*inputs)
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) / inner)
        del output
    return timings


@torch.no_grad()
def kernel_names(
    fn: Callable[..., torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
) -> list[dict[str, Any]]:
    """Capture selected runtime kernel names after timing, outside the gate."""
    try:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            fn(*inputs)
        torch.cuda.synchronize()
        by_name: collections.defaultdict[str, float] = collections.defaultdict(float)
        for event in prof.events():
            if "cuda" not in str(getattr(event, "device_type", "")).lower():
                continue
            value = getattr(event, "self_device_time_total", None)
            if value is None:
                value = getattr(event, "self_cuda_time_total", 0.0)
            by_name[event.name] += float(value)
        return [
            {"name": name, "device_time_us": value}
            for name, value in sorted(by_name.items(), key=lambda item: item[1], reverse=True)[:5]
        ]
    except Exception as exc:  # Profiler proof is useful, but timing must survive profiler drift.
        return [{"profiler_error": repr(exc)}]


def run_benchmark(args: argparse.Namespace, output: Jsonl) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly one visible GPU, got {torch.cuda.device_count()}")
    proof = config_proof(args.variant)
    shapes = tuple(Shape(shape.key, shape.m * args.batch_factor, shape.k, shape.n, shape.roles) for shape in SHAPES)
    expected_tflop_per_step = EXPECTED_TFLOP_PER_STEP * args.batch_factor
    output.emit(
        "config",
        **proof,
        blocks=BLOCKS,
        logical_roles=[role for shape in SHAPES for role in shape.roles],
        logical_shapes=[
            {"role": role, "m": shape.m, "k": shape.k, "n": shape.n}
            for shape in shapes
            for role in shape.roles
        ],
        unique_compile_shapes=[
            {"shape": shape.key, "m": shape.m, "k": shape.k, "n": shape.n, "roles": list(shape.roles)}
            for shape in shapes
        ],
        expected_logical_gemms_per_step=EXPECTED_LOGICAL_GEMMS_PER_STEP,
        expected_tflop_per_step=expected_tflop_per_step,
        batch_factor=args.batch_factor,
        warmup=args.warmup,
        samples=args.samples,
        inner=args.inner,
        seed=args.seed,
    )

    total_weighted_ms = 0.0
    total_compile_s = 0.0
    all_parity_passed = True
    results: list[dict[str, Any]] = []
    for shape_index, shape in enumerate(shapes):
        tensors = make_inputs(shape, args.seed + shape_index)
        for phase in PHASES:
            eager_fn, inputs = phase_call(phase, tensors)
            with torch.no_grad():
                reference = eager_fn(*inputs)

            compiled_fn = torch.compile(eager_fn, fullgraph=True, dynamic=False)
            compile_started = time.monotonic()
            with torch.no_grad():
                candidate = compiled_fn(*inputs)
            torch.cuda.synchronize()
            compile_s = time.monotonic() - compile_started
            parity_stats = parity(candidate, reference)
            del candidate, reference

            timings = time_cuda(compiled_fn, inputs, args.warmup, args.samples, args.inner)
            median_ms = statistics.median(timings)
            weighted_calls = len(shape.roles) * BLOCKS
            weighted_ms = median_ms * weighted_calls
            operation_tflop = 2.0 * shape.m * shape.k * shape.n / 1.0e12
            result = {
                "variant": args.variant,
                "shape": shape.key,
                "roles": list(shape.roles),
                "m": shape.m,
                "k": shape.k,
                "n": shape.n,
                "phase": phase,
                "input_shapes": [list(tensor.shape) for tensor in inputs],
                "input_strides": [list(tensor.stride()) for tensor in inputs],
                "compile_s": compile_s,
                "median_ms": median_ms,
                "min_ms": min(timings),
                "max_ms": max(timings),
                "samples_ms": timings,
                "operation_tflop": operation_tflop,
                "effective_petaflop_s": operation_tflop / median_ms,
                "weighted_calls_per_step": weighted_calls,
                "weighted_step_ms": weighted_ms,
                "parity": parity_stats,
                "selected_runtime_kernels": kernel_names(compiled_fn, inputs),
            }
            output.emit("shape_phase", **result)
            results.append(result)
            total_weighted_ms += weighted_ms
            total_compile_s += compile_s
            all_parity_passed &= parity_stats["passed"]
        del tensors
        torch.cuda.empty_cache()

    output.emit(
        "aggregate",
        variant=args.variant,
        unique_shapes=len(shapes),
        compiled_shape_phases=len(results),
        logical_roles=7,
        logical_gemms_per_step=EXPECTED_LOGICAL_GEMMS_PER_STEP,
        tflop_per_step=expected_tflop_per_step,
        weighted_step_ms=total_weighted_ms,
        effective_petaflop_s=expected_tflop_per_step / total_weighted_ms,
        cold_compile_s=total_compile_s,
        parity_passed=all_parity_passed,
    )
    return 0 if all_parity_passed else 2


def read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def one_record(records: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    matches = [record for record in records if record.get("kind") == kind]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {kind!r} record, found {len(matches)}")
    return matches[0]


def run_compare(args: argparse.Namespace, output: Jsonl) -> int:
    assert args.compare is not None
    paths = args.compare
    record_sets = [read_records(path) for path in paths]
    aggregates = [one_record(records, "aggregate") for records in record_sets]
    labels = ("current_a", "cutlass", "current_b")
    for expected, aggregate in zip(("current", "cutlass", "current"), aggregates):
        if aggregate["variant"] != expected:
            raise RuntimeError(f"expected {expected}, got {aggregate['variant']}")
    tflop_per_step = aggregates[0]["tflop_per_step"]
    if any(aggregate["tflop_per_step"] != tflop_per_step for aggregate in aggregates[1:]):
        raise RuntimeError("A/X/B FLOP counts differ")

    baseline_ms = statistics.mean((aggregates[0]["weighted_step_ms"], aggregates[2]["weighted_step_ms"]))
    candidate_ms = aggregates[1]["weighted_step_ms"]
    saving_ms = baseline_ms - candidate_ms
    output.emit(
        "comparison",
        inputs={label: str(path) for label, path in zip(labels, paths)},
        weighted_step_ms={label: aggregate["weighted_step_ms"] for label, aggregate in zip(labels, aggregates)},
        baseline_midpoint_ms=baseline_ms,
        candidate_ms=candidate_ms,
        saving_ms=saving_ms,
        saving_percent=100.0 * saving_ms / baseline_ms,
        control_spread_ms=abs(aggregates[0]["weighted_step_ms"] - aggregates[2]["weighted_step_ms"]),
        baseline_effective_petaflop_s=tflop_per_step / baseline_ms,
        candidate_effective_petaflop_s=tflop_per_step / candidate_ms,
        all_parity_passed=all(aggregate["parity_passed"] for aggregate in aggregates),
        pass_two_ms_gate=saving_ms >= 2.0,
    )

    keyed = []
    for records in record_sets:
        keyed.append({(r["shape"], r["phase"]): r for r in records if r.get("kind") == "shape_phase"})
    if keyed[0].keys() != keyed[1].keys() or keyed[0].keys() != keyed[2].keys():
        raise RuntimeError("shape/phase sets differ")
    for key in sorted(keyed[0]):
        a, candidate, b = (records[key] for records in keyed)
        base = statistics.mean((a["weighted_step_ms"], b["weighted_step_ms"]))
        output.emit(
            "comparison_shape_phase",
            shape=key[0],
            phase=key[1],
            baseline_midpoint_ms=base,
            candidate_ms=candidate["weighted_step_ms"],
            saving_ms=base - candidate["weighted_step_ms"],
            saving_percent=100.0 * (base - candidate["weighted_step_ms"]) / base,
            candidate_selected_runtime_kernels=candidate["selected_runtime_kernels"],
        )
    return 0


def main() -> int:
    args = parse_args()
    output = Jsonl(args.output)
    try:
        if args.compare is not None:
            return run_compare(args, output)
        return run_benchmark(args, output)
    finally:
        output.close()


if __name__ == "__main__":
    sys.exit(main())
