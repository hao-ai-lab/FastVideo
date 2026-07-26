#!/usr/bin/env python3
"""FA4 forward-efficiency tail-tile discriminator (plan item 1).

The trainer profile shows FA4 forward at 34.9% of peak while backward runs
46.2% — inverted vs every flash-attention generation. One candidate mechanism
is the ragged sequence: S=4,290 is 33.5 tiles of 128. This probe measures
per-FLOP throughput of the production FA4 path at S in {4,224, 4,290, 4,352}
(clean / ragged / clean tile counts), forward-only and forward+backward, at
B2 and B3. If the clean neighbors run >2-3% faster per FLOP than 4,290 in
forward, the tail hypothesis is confirmed and a seqused/padded integration is
worth pursuing; if per-FLOP throughput is flat, the anomaly lives in schedule
or occupancy instead. Ratios only; degraded-bin trays are acceptable.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

HEADS, HEAD_DIM = 32, 128


def time_fn(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(repeats)]
    for start, end in events:
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(start.elapsed_time(end) for start, end in events)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=31)
    args = parser.parse_args()

    from fastvideo.attention.utils.flash_attn_cute import flash_attn_func as fa4

    torch.manual_seed(20260722)
    device = torch.device("cuda:0")
    rows = []
    for batch in (2, 3):
        for tokens in (4224, 4290, 4352):
            shape = (batch, tokens, HEADS, HEAD_DIM)
            q = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True)
            grad_out = torch.randn(shape, device=device, dtype=torch.bfloat16)

            def fwd_only():
                with torch.no_grad():
                    fa4(q, k, v)

            def fwd_bwd():
                q.grad = k.grad = v.grad = None
                fa4(q, k, v).backward(grad_out)

            fwd_ms = time_fn(fwd_only, args.warmup, args.repeats)
            total_ms = time_fn(fwd_bwd, args.warmup, args.repeats)
            d_total = HEADS * HEAD_DIM
            fwd_tf = 4.0 * batch * tokens * tokens * d_total / 1e12
            bwd_tf = 10.0 * batch * tokens * tokens * d_total / 1e12
            row = {
                "batch": batch,
                "tokens": tokens,
                "tiles_128": tokens / 128.0,
                "fwd_ms": round(fwd_ms, 4),
                "fwd_tflops_per_s": round(fwd_tf / (fwd_ms / 1000.0), 1),
                "fwd_bwd_ms": round(total_ms, 4),
                "bwd_ms_est": round(total_ms - fwd_ms, 4),
                "bwd_tflops_per_s_est": round(bwd_tf / ((total_ms - fwd_ms) / 1000.0), 1),
            }
            rows.append(row)
            print("FA4_TAIL_CASE " + json.dumps(row, sort_keys=True), flush=True)
            del q, k, v, grad_out
            torch.cuda.empty_cache()

    for batch in (2, 3):
        sub = {r["tokens"]: r for r in rows if r["batch"] == batch}
        ragged = sub[4290]["fwd_tflops_per_s"]
        clean_low, clean_high = sub[4224]["fwd_tflops_per_s"], sub[4352]["fwd_tflops_per_s"]
        print(
            "FA4_TAIL_VERDICT " + json.dumps({
                "batch": batch,
                "fwd_ragged_tflops": ragged,
                "fwd_clean_low_tflops": clean_low,
                "fwd_clean_high_tflops": clean_high,
                "clean_over_ragged_pct": round(
                    100.0 * (max(clean_low, clean_high) - ragged) / ragged, 2),
            }, sort_keys=True),
            flush=True,
        )


if __name__ == "__main__":
    main()
