#!/usr/bin/env python3
"""Exact-shape LTX-2 BF16 GELU-epilogue microbenchmark for GB200."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
import fused_dense_lib


BLOCKS = 48
DTYPE = torch.bfloat16


def _reference_forward(
    x: torch.Tensor,
    weight1: torch.Tensor,
    bias1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pre_activation = F.linear(x, weight1, bias1)
    return F.gelu(pre_activation, approximate="tanh"), pre_activation


def _reference_backward(
    grad_output: torch.Tensor,
    weight2: torch.Tensor,
    pre_activation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_activation = F.linear(grad_output, weight2.t())
    grad_pre_activation = torch.ops.aten.gelu_backward.default(
        grad_activation,
        pre_activation,
        approximate="tanh",
    )
    return grad_pre_activation, grad_pre_activation.sum(dim=0)


reference_forward = torch.compile(_reference_forward, fullgraph=True, dynamic=False)
reference_backward = torch.compile(_reference_backward, fullgraph=True, dynamic=False)


def _time_ms(fn: Callable[[], Any], iterations: int) -> float:
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / iterations


def _peak_memory(fn: Callable[[], Any]) -> dict[str, float]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    base_allocated = torch.cuda.memory_allocated()
    base_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    outputs = fn()
    torch.cuda.synchronize()
    result = {
        "allocated_delta_mib": (torch.cuda.max_memory_allocated() - base_allocated) / 2**20,
        "reserved_delta_mib": (torch.cuda.max_memory_reserved() - base_reserved) / 2**20,
        "live_output_delta_mib": (torch.cuda.memory_allocated() - base_allocated) / 2**20,
    }
    del outputs
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
) -> float:
    if actual.dtype != DTYPE or expected.dtype != DTYPE:
        raise AssertionError(f"{name}: expected BF16 tensors, got {actual.dtype} and {expected.dtype}")
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=atol)
    return float((actual - expected).abs().max())


def _run_shape(
    m: int,
    *,
    warmup: int,
    iterations: int,
    rounds: int,
    heuristic: int,
    seed: int,
) -> dict[str, Any]:
    d, h = 4096, 16384
    torch.manual_seed(seed + m)
    x = torch.empty((m, d), device="cuda", dtype=DTYPE).normal_(std=1.0)
    weight1 = torch.empty((h, d), device="cuda", dtype=DTYPE).normal_(std=0.02)
    bias1 = torch.empty((h,), device="cuda", dtype=DTYPE).normal_(std=0.01)
    weight2 = torch.empty((d, h), device="cuda", dtype=DTYPE).normal_(std=0.02)
    grad_output = torch.empty((m, d), device="cuda", dtype=DTYPE).normal_(std=1 / 32)

    def reference() -> tuple[torch.Tensor, ...]:
        activation, pre_activation = reference_forward(x, weight1, bias1)
        grad_pre_activation, grad_bias1 = reference_backward(grad_output, weight2, pre_activation)
        return activation, pre_activation, grad_pre_activation, grad_bias1

    def fused() -> tuple[torch.Tensor, ...]:
        activation, pre_activation = fused_dense_lib.linear_act_forward(
            x,
            weight1,
            bias1,
            True,
            True,
            heuristic,
        )
        grad_pre_activation, grad_bias1 = fused_dense_lib.bias_act_linear_dgrad_bgrad(
            weight2,
            grad_output,
            pre_activation,
            True,
            heuristic,
        )
        return activation, pre_activation, grad_pre_activation, grad_bias1

    reference_outputs = reference()
    fused_outputs = fused()
    names = ("activation", "pre_activation", "grad_pre_activation", "grad_bias1")
    atols = (3e-2, 3e-2, 3e-2, 1.5e-1)
    parity_max_abs = {
        name: _assert_close(name, actual, expected, atol=atol)
        for name, actual, expected, atol in zip(names, fused_outputs, reference_outputs, atols)
    }
    del reference_outputs, fused_outputs
    gc.collect()
    torch.cuda.empty_cache()

    for _ in range(warmup):
        reference()
        fused()
    torch.cuda.synchronize()

    samples: dict[str, list[float]] = {"reference": [], "fused": []}
    functions = {"reference": reference, "fused": fused}
    for round_index in range(rounds):
        order = ("reference", "fused") if round_index % 2 == 0 else ("fused", "reference")
        for name in order:
            samples[name].append(_time_ms(functions[name], iterations))

    medians = {name: statistics.median(values) for name, values in samples.items()}
    saving_ms = medians["reference"] - medians["fused"]
    peak_memory = {name: _peak_memory(functions[name]) for name in ("reference", "fused")}
    return {
        "m": m,
        "d": d,
        "h": h,
        "dtype": str(DTYPE),
        "heuristic": heuristic,
        "reference_median_ms": medians["reference"],
        "fused_median_ms": medians["fused"],
        "per_block_saving_ms": saving_ms,
        "projected_48_block_saving_ms": BLOCKS * saving_ms,
        "isolated_speedup_percent": 100 * saving_ms / medians["reference"],
        "samples_ms": samples,
        "parity_max_abs": parity_max_abs,
        "peak_memory": peak_memory,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", nargs="+", type=int, default=[4290, 12870])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--heuristic", type=int, default=0, choices=range(5))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.m) <= 0 or min(args.warmup, args.iterations, args.rounds) <= 0:
        parser.error("shapes and measurement counts must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 10:
        raise RuntimeError(f"expected GB200/SM100, got compute capability {capability}")

    torch.set_grad_enabled(False)
    print(
        "LTX2_FUSED_DENSE_GELU_ENV "
        + json.dumps(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(),
                "capability": capability,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for m in args.m:
        result = _run_shape(
            m,
            warmup=args.warmup,
            iterations=args.iterations,
            rounds=args.rounds,
            heuristic=args.heuristic,
            seed=args.seed,
        )
        print("LTX2_FUSED_DENSE_GELU_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
