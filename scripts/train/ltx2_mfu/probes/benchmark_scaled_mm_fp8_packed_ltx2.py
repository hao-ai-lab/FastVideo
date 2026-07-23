#!/usr/bin/env python3
"""Scratch exact-shape prequantized FP8 ceiling for LTX-2 packed linears."""

import gc
import json
import statistics

import torch
import torch.nn.functional as F


LAYERS = (
    ("self_qkv", 48, 4290, 4096, 12288),
    ("video_dd", 144, 4290, 4096, 4096),
    ("text_kv", 48, 1024, 4096, 8192),
    ("ffn_up", 48, 4290, 4096, 16384),
    ("ffn_down", 48, 4290, 16384, 4096),
)


def timed(fn, warmup=5, samples=9, inner=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(samples):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(inner):
            fn()
        end.record()
        end.synchronize()
        values.append(begin.elapsed_time(end) / inner)
    return statistics.median(values)


def operands(m, k, n, phase):
    if phase == "fwd":
        return (torch.randn(m, k, device="cuda", dtype=torch.bfloat16),
                torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02)
    if phase == "dgrad":
        return (torch.randn(m, n, device="cuda", dtype=torch.bfloat16) * 0.01,
                (torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02).T.contiguous())
    dy = torch.randn(m, n, device="cuda", dtype=torch.bfloat16) * 0.01
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    padded = (m + 31) // 32 * 32
    if padded != m:
        dy = F.pad(dy, (0, 0, 0, padded - m))
        x = F.pad(x, (0, 0, 0, padded - m))
    return dy.T.contiguous(), x.T.contiguous()


def bench(name, count, m, k, n, phase):
    lhs, rhs_rows = operands(m, k, n, phase)
    lhs8 = lhs.to(torch.float8_e4m3fn)
    rhs8 = rhs_rows.to(torch.float8_e4m3fn)
    scale = torch.ones((), device="cuda", dtype=torch.float32)

    def fp8(fast):
        return torch._scaled_mm(
            lhs8,
            rhs8.T,
            scale_a=scale,
            scale_b=scale,
            out_dtype=torch.bfloat16,
            use_fast_accum=fast,
        )

    reference = lhs @ rhs_rows.T
    actual = fp8(True)
    if isinstance(actual, tuple):
        actual = actual[0]
    torch.cuda.synchronize()
    diff = actual.float() - reference.float()
    row = {
        "kind": "case",
        "name": name,
        "phase": phase,
        "count": count,
        "logical_shape": [m, k, n],
        "qmm_shape": [lhs.shape[0], lhs.shape[1], rhs_rows.shape[0]],
        "bf16_ms": timed(lambda: lhs @ rhs_rows.T),
        "fp8_fast_ms": timed(lambda: fp8(True)),
        "fp8_accurate_ms": timed(lambda: fp8(False)),
        "relative_rms": float(diff.square().mean().sqrt() /
                              reference.float().square().mean().sqrt()),
    }
    print(json.dumps(row, sort_keys=True), flush=True)
    return row


if __name__ == "__main__":
    assert torch.cuda.get_device_capability() == (10, 0)
    torch.manual_seed(20260721)
    rows = []
    for layer in LAYERS:
        for phase in ("fwd", "dgrad", "wgrad"):
            rows.append(bench(*layer, phase))
            gc.collect()
            torch.cuda.empty_cache()
    totals = {
        tier: sum(row[f"{tier}_ms"] * row["count"] for row in rows)
        for tier in ("bf16", "fp8_fast", "fp8_accurate")
    }
    print(json.dumps({
        "kind": "aggregate",
        "totals_ms_48_blocks": totals,
        "speedup_fast": totals["bf16"] / totals["fp8_fast"],
        "speedup_accurate": totals["bf16"] / totals["fp8_accurate"],
    }, sort_keys=True))
