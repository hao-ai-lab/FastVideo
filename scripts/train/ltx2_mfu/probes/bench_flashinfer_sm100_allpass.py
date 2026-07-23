#!/usr/bin/env python3
"""Exact seven-pack LTX-2 all-pass lower bound with installed SM100 CuTe kernels."""

import gc
import json
import statistics

import torch
import torch.nn.functional as F
from flashinfer import fp4_quantize, mm_fp4


LAYERS = (
    ("self_qkv", 48, 4290, 4096, 12288),
    ("video_dd", 144, 4290, 4096, 4096),
    ("text_kv", 48, 1024, 4096, 8192),
    ("ffn_up", 48, 4290, 4096, 16384),
    ("ffn_down", 48, 4290, 16384, 4096),
)


def timed(fn, warmup=3, samples=5, inner=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(samples):
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        begin.record()
        for _ in range(inner):
            fn()
        end.record()
        end.synchronize()
        values.append(begin.elapsed_time(end) / inner)
    return statistics.median(values)


def global_scale(x):
    return (448.0 * 6.0) / x.float().abs().nan_to_num().amax().clamp_min(1e-12)


def quant(x, scale):
    return fp4_quantize(x, scale, backend="cute-dsl", enable_pdl=True)


def operands(m, k, n, phase):
    if phase == "fwd":
        return (torch.randn(m, k, device="cuda", dtype=torch.bfloat16),
                torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02, True)
    if phase == "dgrad":
        return (torch.randn(m, n, device="cuda", dtype=torch.bfloat16) * 0.01,
                (torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02).T.contiguous(), True)
    dy = torch.randn(m, n, device="cuda", dtype=torch.bfloat16) * 0.01
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    padded = (m + 31) // 32 * 32
    if padded != m:
        dy, x = F.pad(dy, (0, 0, 0, padded - m)), F.pad(x, (0, 0, 0, padded - m))
    return dy.T.contiguous(), x.T.contiguous(), False


def bench(name, count, m, k, n, phase):
    lhs, rhs_rows, rhs_is_weight = operands(m, k, n, phase)
    ls, rs = global_scale(lhs), global_scale(rhs_rows)
    lq, lsf = quant(lhs, ls)
    rq, rsf = quant(rhs_rows, rs)
    alpha = (ls * rs).reciprocal()
    out = torch.empty(lhs.shape[0], rhs_rows.shape[0], device="cuda", dtype=torch.bfloat16)

    def mm():
        return mm_fp4(lq, rq.T, lsf, rsf.T, alpha, out=out, backend="cute-dsl", enable_pdl=True)

    def delayed():
        aq, asf = quant(lhs, ls)
        if rhs_is_weight:
            bq, bsf = rq, rsf
        else:
            bq, bsf = quant(rhs_rows, rs)
        return mm_fp4(aq, bq.T, asf, bsf.T, alpha, out=out, backend="cute-dsl", enable_pdl=True)

    def exact():
        als, brs = global_scale(lhs), global_scale(rhs_rows)
        aq, asf = quant(lhs, als)
        if rhs_is_weight:
            bq, bsf = rq, rsf
            brs = rs
        else:
            bq, bsf = quant(rhs_rows, brs)
        return mm_fp4(aq, bq.T, asf, bsf.T, (als * brs).reciprocal(), out=out,
                      backend="cute-dsl", enable_pdl=True)

    reference = lhs @ rhs_rows.T
    actual = mm().clone()
    torch.cuda.synchronize()
    diff = actual.float() - reference.float()
    row = {
        "name": name, "phase": phase, "count": count,
        "logical_shape": [m, k, n], "qmm_shape": [lhs.shape[0], lhs.shape[1], rhs_rows.shape[0]],
        "bf16_ms": timed(lambda: torch.mm(lhs, rhs_rows.T, out=out)),
        "prequant_ms": timed(mm), "delayed_ms": timed(delayed), "exact_ms": timed(exact),
        "rhs_refresh_delayed_ms": timed(lambda: quant(rhs_rows, rs), inner=1) if rhs_is_weight else 0.0,
        "rhs_refresh_exact_ms": timed(lambda: quant(rhs_rows, global_scale(rhs_rows)), inner=1) if rhs_is_weight else 0.0,
        "relative_rms": float(diff.square().mean().sqrt() / reference.float().square().mean().sqrt()),
    }
    print(json.dumps({"kind": "case", **row}, sort_keys=True), flush=True)
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
    totals = {}
    for tier in ("bf16", "prequant", "delayed", "exact"):
        value = sum(r[f"{tier}_ms"] * r["count"] for r in rows)
        if tier == "delayed":
            value += sum(r["rhs_refresh_delayed_ms"] * r["count"] for r in rows)
        elif tier == "exact":
            value += sum(r["rhs_refresh_exact_ms"] * r["count"] for r in rows)
        totals[tier] = {"weighted_ms_48_blocks": value, "ms_per_block": value / 48}
    baseline = 198.4054238319397
    print(json.dumps({"kind": "aggregate", "baseline_separate_bf16_ms": baseline,
                      "target_ms": baseline / 2.5, "robust_target_ms": baseline / 2.7,
                      "totals": totals}, sort_keys=True))
