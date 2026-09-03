# SPDX-License-Identifier: Apache-2.0
"""Benchmark the coarse/sparse combine in the 64-block VSA path.

Compares the previous implementation (coarse output expanded across the full
sequence with ``repeat()``, then an out-of-place multiply and add) against the
current one (coarse output kept at block resolution and broadcast).

  python benchmarks/bench_vsa_combine.py            # CI-sized shapes
  python benchmarks/bench_vsa_combine.py --large    # adds an H3-like shape

The large case is representative of MiniMax H3 inference: bf16, 56 heads, head
dim 128, ~15-16k tokens, block size 64, top-k ~20%. It needs roughly 3 GiB and
is opt-in so the default run stays cheap.
"""
from __future__ import annotations

import argparse

import torch

from fastvideo_kernel.ops import _combine_coarse_sparse

BE = 64


def _reference_combine(out_c, out_s, weight, batch, heads, n_blocks, be, dim, seq):
    out_c = out_c.repeat(1, 1, 1, be, 1).view(batch, heads, seq, dim)
    if weight is not None:
        return out_c * weight + out_s
    return out_c + out_s


def _time(fn, iters: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _measure(fn, iters):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_a, base_r = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    ms = _time(fn, iters)
    torch.cuda.synchronize()
    return (ms,
            (torch.cuda.max_memory_allocated() - base_a) / 2**20,
            (torch.cuda.max_memory_reserved() - base_r) / 2**20)


def run_case(name, batch, heads, n_blocks, dim, gated, iters):
    seq = n_blocks * BE
    torch.manual_seed(0)
    out_c = torch.randn(batch, heads, n_blocks, 1, dim, device="cuda", dtype=torch.bfloat16)
    out_s = torch.randn(batch, heads, seq, dim, device="cuda", dtype=torch.bfloat16)
    weight = torch.rand(batch, heads, seq, dim, device="cuda", dtype=torch.bfloat16) if gated else None
    full_mib = batch * heads * seq * dim * 2 / 2**20

    with torch.no_grad():
        ref_ms, ref_a, ref_r = _measure(
            lambda: _reference_combine(out_c, out_s.clone(), weight, batch, heads, n_blocks, BE, dim,
                                       seq), iters)
        new_ms, new_a, new_r = _measure(
            lambda: _combine_coarse_sparse(out_c, out_s.clone(), weight, batch, heads, n_blocks, BE,
                                           dim, seq), iters)

    print(f"\n{name}   B={batch} H={heads} S={seq} D={dim} gated={gated}  "
          f"(one full [B,H,S,D] bf16 tensor = {full_mib:.1f} MiB)")
    print(f"  {'':<8}{'latency ms':>12}{'peak alloc MiB':>17}{'peak reserved MiB':>20}")
    print(f"  {'before':<8}{ref_ms:>12.4f}{ref_a:>17.1f}{ref_r:>20.1f}")
    print(f"  {'after':<8}{new_ms:>12.4f}{new_a:>17.1f}{new_r:>20.1f}")
    speed = (ref_ms / new_ms - 1.0) * 100.0 if new_ms else 0.0
    print(f"  {'delta':<8}{speed:>11.1f}%{new_a - ref_a:>17.1f}{new_r - ref_r:>20.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--large", action="store_true", help="add the H3-like ~15.4k-token case")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm device required")
    print("device:", torch.cuda.get_device_name(0))

    for gated in (False, True):
        run_case("small ", 1, 8, 64, 128, gated, args.iters)
    if args.large:
        # 15,488 tokens = 242 blocks x 64, matching MiniMax H3 at 864x480/124f
        for gated in (False, True):
            run_case("H3-like", 1, 56, 242, 128, gated, max(10, args.iters // 5))


if __name__ == "__main__":
    main()
