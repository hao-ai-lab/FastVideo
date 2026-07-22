#!/usr/bin/env python3
"""Exact LTX-2 B=1 fprop/dgrad/wgrad benchmark for TorchAO NVFP4."""

import argparse
import gc
import json
import statistics
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor,
    _addmm_nvfp4_dispatch,
    per_tensor_amax_to_scale,
)


@dataclass(frozen=True)
class Layer:
    name: str
    count: int
    m: int
    k: int
    n: int


LAYOUTS = {
    "separate": (
        Layer("video_dd", 288, 4290, 4096, 4096),
        Layer("text_dd", 96, 1024, 4096, 4096),
        Layer("ffn_up", 48, 4290, 4096, 16384),
        Layer("ffn_down", 48, 4290, 16384, 4096),
    ),
    "packed": (
        Layer("self_qkv", 48, 4290, 4096, 12288),
        Layer("video_dd", 144, 4290, 4096, 4096),
        Layer("text_kv", 48, 1024, 4096, 8192),
        Layer("ffn_up", 48, 4290, 4096, 16384),
        Layer("ffn_down", 48, 4290, 16384, 4096),
    ),
}
PHASES = ("fwd", "dgrad", "wgrad")
TIERS = (
    "bf16",
    "fp4_mm",
    "fp4_mm_2level",
    "fp4_eager",
    "fp4_eager_dynamic",
    "fp4_triton_static",
    "fp4_triton_dynamic",
)


def emit(kind: str, **values: object) -> None:
    print(json.dumps({"kind": kind, **values}, sort_keys=True), flush=True)


def timing(fn: Callable[[], torch.Tensor], warmup: int, samples: int, inner: int) -> dict[str, float]:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        values = []
        for _ in range(samples):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(inner):
                fn()
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end) / inner)
    return {"median_ms": statistics.median(values), "min_ms": min(values), "max_ms": max(values)}


def error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    actual = actual.float()
    expected = expected.float()
    diff = actual - expected
    return {
        "finite": bool(torch.isfinite(actual).all().item()),
        "relative_rms": (diff.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-30)).item(),
        "mean_abs": diff.abs().mean().item(),
        "max_abs": diff.abs().amax().item(),
    }


def make_case(layer: Layer, phase: str):
    m, k, n = layer.m, layer.k, layer.n
    if phase == "fwd":
        x = torch.empty((m, k), device="cuda", dtype=torch.bfloat16).normal_()
        w = torch.empty((n, k), device="cuda", dtype=torch.bfloat16).normal_(0, 0.02)
        return lambda: torch.mm(x, w.T), lambda: (x, w), (m, k, n), 2 * m * k * n
    if phase == "dgrad":
        dy = torch.empty((m, n), device="cuda", dtype=torch.bfloat16).normal_(0, 0.01)
        w = torch.empty((n, k), device="cuda", dtype=torch.bfloat16).normal_(0, 0.02)
        return lambda: torch.mm(dy, w), lambda: (dy, w.T.contiguous()), (m, n, k), 2 * m * n * k
    if phase == "wgrad":
        dy = torch.empty((m, n), device="cuda", dtype=torch.bfloat16).normal_(0, 0.01)
        x = torch.empty((m, k), device="cuda", dtype=torch.bfloat16).normal_()
        padded_m = (m + 31) // 32 * 32

        def orient():
            pad = padded_m - m
            dy_padded = F.pad(dy, (0, 0, 0, pad)) if pad else dy
            x_padded = F.pad(x, (0, 0, 0, pad)) if pad else x
            return dy_padded.T.contiguous(), x_padded.T.contiguous()

        return lambda: torch.mm(dy.T, x), orient, (n, padded_m, k), 2 * m * n * k
    raise ValueError(phase)


def quantize(tensor: torch.Tensor, triton: bool, dynamic: bool, static_scale: torch.Tensor | None):
    scale = per_tensor_amax_to_scale(tensor.abs().amax()) if dynamic else static_scale
    return NVFP4Tensor.to_nvfp4(
        tensor,
        per_tensor_scale=scale,
        is_swizzled_scales=True,
        use_triton_kernel=triton,
    )


def mm(lhs: NVFP4Tensor, rhs_rows: NVFP4Tensor) -> torch.Tensor:
    return _addmm_nvfp4_dispatch(lhs, rhs_rows.t(), None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--inner", type=int, default=3)
    parser.add_argument("--layouts", nargs="+", default=list(LAYOUTS))
    parser.add_argument("--phases", nargs="+", default=list(PHASES))
    parser.add_argument("--tiers", nargs="+", default=list(TIERS))
    parser.add_argument("--concat", action="store_true")
    args = parser.parse_args()

    torch.cuda.set_device(0)
    torch.manual_seed(20260721)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    emit(
        "environment",
        torch=torch.__version__,
        torchao=__import__("torchao").__version__,
        gpu=torch.cuda.get_device_name(0),
        capability=torch.cuda.get_device_capability(0),
        warmup=args.warmup,
        samples=args.samples,
        inner=args.inner,
    )
    aggregates = {
        layout: {tier: {phase: {"latency_ms": 0.0, "logical_flops": 0.0} for phase in PHASES} for tier in args.tiers}
        for layout in args.layouts
    }

    for layout in args.layouts:
      for phase in args.phases:
        for index, layer in enumerate(LAYOUTS[layout]):
            gc.collect()
            torch.cuda.empty_cache()
            torch.manual_seed(20260721 + 100 * index + PHASES.index(phase))
            bf16, orient, qshape, logical_flops = make_case(layer, phase)
            reference = bf16()
            lhs, rhs = orient()
            static_lhs_scale = per_tensor_amax_to_scale(lhs.abs().amax())
            static_rhs_scale = per_tensor_amax_to_scale(rhs.abs().amax())
            lhs_q = quantize(lhs, False, False, None)
            rhs_q = quantize(rhs, False, False, None)
            lhs_q_2level = quantize(lhs, False, False, static_lhs_scale)
            rhs_q_2level = quantize(rhs, False, False, static_rhs_scale)
            torch.cuda.synchronize()

            calls = {
                "bf16": bf16,
                "fp4_mm": lambda: mm(lhs_q, rhs_q),
                "fp4_mm_2level": lambda: mm(lhs_q_2level, rhs_q_2level),
            }

            def eager():
                a, b = orient()
                return mm(quantize(a, False, False, None), quantize(b, False, False, None))

            def triton_static():
                a, b = orient()
                return mm(
                    quantize(a, True, False, static_lhs_scale),
                    quantize(b, True, False, static_rhs_scale),
                )

            def triton_dynamic():
                a, b = orient()
                return mm(quantize(a, True, True, None), quantize(b, True, True, None))

            def eager_dynamic():
                a, b = orient()
                return mm(quantize(a, False, True, None), quantize(b, False, True, None))

            calls.update(
                fp4_eager=eager,
                fp4_eager_dynamic=eager_dynamic,
                fp4_triton_static=triton_static,
                fp4_triton_dynamic=triton_dynamic,
            )
            for tier in args.tiers:
                try:
                    result = calls[tier]()
                    torch.cuda.synchronize()
                    metrics = timing(calls[tier], args.warmup, args.samples, args.inner)
                    numerical = None if tier == "bf16" else error(result, reference)
                    aggregates[layout][tier][phase]["latency_ms"] += metrics["median_ms"] * layer.count
                    aggregates[layout][tier][phase]["logical_flops"] += logical_flops * layer.count
                    emit(
                        "case",
                        layout=layout,
                        tier=tier,
                        phase=phase,
                        layer=layer.name,
                        count=layer.count,
                        logical_shape=(layer.m, layer.k, layer.n),
                        qmm_shape=qshape,
                        logical_flops=logical_flops,
                        error=numerical,
                        **metrics,
                    )
                except Exception as exc:
                    torch.cuda.synchronize()
                    emit("failure", tier=tier, phase=phase, layer=layer.name, qmm_shape=qshape, exception=repr(exc))

            del reference, lhs, rhs, lhs_q, rhs_q, lhs_q_2level, rhs_q_2level

    totals = {}
    for layout, tiers in aggregates.items():
        totals[layout] = {}
        for tier, phases in tiers.items():
            latency = sum(row["latency_ms"] for row in phases.values())
            flops = sum(row["logical_flops"] for row in phases.values())
            complete = all(row["logical_flops"] > 0 for row in phases.values())
            totals[layout][tier] = {
                "complete": complete,
                "total_latency_ms": latency,
                "total_logical_flops": flops,
                "effective_tflops": flops / latency / 1e9 if latency else 0.0,
                "phases": phases,
            }
            emit("aggregate", layout=layout, tier=tier, **totals[layout][tier])
    if "separate" in totals and "bf16" in totals["separate"]:
        bf16_ms = totals["separate"]["bf16"]["total_latency_ms"]
        candidates = {
            f"{layout}/{tier}": row
            for layout, tiers in totals.items()
            for tier, row in tiers.items()
            if not (layout == "separate" and tier == "bf16") and row["total_latency_ms"]
        }
        emit(
            "gate",
            baseline="separate/bf16",
            required_speedup=2.5,
            results={name: {"speedup_vs_separate_bf16": bf16_ms / row["total_latency_ms"], "passes": row["complete"] and bf16_ms / row["total_latency_ms"] > 2.5} for name, row in candidates.items()},
        )
    if args.concat:
        concat_total = 0.0
        for name, shape, copies, dim in (
            ("self_qkv_weight", (4096, 4096), 3, 0),
            ("text_kv_weight", (4096, 4096), 2, 0),
            ("self_qkv_grad_output", (4290, 4096), 3, 1),
            ("text_kv_grad_output", (1024, 4096), 2, 1),
        ):
            inputs = [torch.empty(shape, device="cuda", dtype=torch.bfloat16).normal_() for _ in range(copies)]
            metrics = timing(lambda: torch.cat(inputs, dim=dim), args.warmup, args.samples, args.inner)
            weighted_ms = metrics["median_ms"] * 48
            concat_total += weighted_ms
            emit("concat", name=name, count=48, weighted_ms=weighted_ms, **metrics)
            del inputs
            torch.cuda.empty_cache()
        emit("concat_aggregate", weighted_ms=concat_total)
    emit("done", elapsed_wall_seconds=time.monotonic() - started)


if __name__ == "__main__":
    main()
