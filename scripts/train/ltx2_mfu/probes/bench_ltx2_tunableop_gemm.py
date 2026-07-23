#!/usr/bin/env python3
"""Occurrence-weighted packed LTX-2 GEMM band probe for TunableOp / env gates.

Runs the five packed projection cases as real ``torch.nn.Linear`` fwd+bwd so
cuBLAS sees the production orientations (fwd addmm, dgrad NN, wgrad TN with
bias grad), weights each case by its per-step occurrence count (48 blocks,
video_dd three per block), and prints a per-step GEMM band estimate. Compare
one process without TunableOp against a tune-then-replay pair with
``PYTORCH_TUNABLEOP_ENABLED=1``. Never MFU evidence.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch
import torch.nn.functional as F

HIDDEN = 4096
FFN = 16384
VIDEO_TOKENS = 11 * 15 * 26
TEXT_TOKENS = 1024


def build_cases(batch: int) -> list[dict]:
    video_rows = VIDEO_TOKENS * batch
    text_rows = TEXT_TOKENS * batch
    return [
        {"name": "self_qkv", "rows": video_rows, "in": HIDDEN, "out": 3 * HIDDEN, "occurrences": 48},
        {"name": "video_dd", "rows": video_rows, "in": HIDDEN, "out": HIDDEN, "occurrences": 144},
        {"name": "text_kv", "rows": text_rows, "in": HIDDEN, "out": 2 * HIDDEN, "occurrences": 48},
        {"name": "ffn_up", "rows": video_rows, "in": HIDDEN, "out": FFN, "occurrences": 48},
        {"name": "ffn_down", "rows": video_rows, "in": FFN, "out": HIDDEN, "occurrences": 48},
    ]


def time_case(case: dict, warmup: int, repeats: int) -> dict:
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    layer = torch.nn.Linear(case["in"], case["out"], bias=True, device=device, dtype=dtype)
    x = torch.randn(case["rows"], case["in"], device=device, dtype=dtype, requires_grad=True)
    grad_out = torch.randn(case["rows"], case["out"], device=device, dtype=dtype)

    def step() -> None:
        x.grad = None
        layer.weight.grad = None
        layer.bias.grad = None
        F.linear(x, layer.weight, layer.bias).backward(grad_out)

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        step()
        end.record()
    torch.cuda.synchronize()
    values = [float(start.elapsed_time(end)) for start, end in zip(starts, ends, strict=True)]
    median_ms = statistics.median(values)
    flops = 6.0 * case["rows"] * case["in"] * case["out"]
    return {
        "case": case["name"],
        "rows": case["rows"],
        "median_ms": median_ms,
        "min_ms": min(values),
        "occurrences": case["occurrences"],
        "weighted_ms": median_ms * case["occurrences"],
        "tflops": flops / (median_ms * 1e9),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--label", default="default")
    args = parser.parse_args()

    torch.manual_seed(20260722)
    rows = [time_case(case, args.warmup, args.repeats) for case in build_cases(args.batch)]
    total_weighted = sum(row["weighted_ms"] for row in rows)
    total_flops = sum(
        6.0 * case["rows"] * case["in"] * case["out"] * case["occurrences"]
        for case in build_cases(args.batch))
    print(
        "GEMM_BAND " + json.dumps({
            "label": args.label,
            "batch": args.batch,
            "tunableop_enabled": torch.cuda.tunable.is_enabled(),
            "tunableop_tuning": torch.cuda.tunable.tuning_is_enabled(),
            "per_step_gemm_band_ms": round(total_weighted, 3),
            "band_tflops": round(total_flops / (total_weighted * 1e9), 1),
            "cases": rows,
        }, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
