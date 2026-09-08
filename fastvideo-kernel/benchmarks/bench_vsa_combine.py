# SPDX-License-Identifier: Apache-2.0
"""Benchmark the coarse/sparse combine used by ``video_sparse_attn`` (BHSD entry, every tile size).

Compares the previous implementation (coarse output expanded across the full
sequence with ``repeat()``, then an out-of-place multiply and add: three
full-sequence temporaries) against the current one (block-resolution broadcast,
fused ``addcmul``, in place whenever the sparse output is not in an autograd
graph, one temporary otherwise).

  python benchmarks/bench_vsa_combine.py             # combine alone, CI-sized shape
  python benchmarks/bench_vsa_combine.py --large     # adds 56x15488x128 and a Wan-14B 480p-like shape
  python benchmarks/bench_vsa_combine.py --e2e       # full video_sparse_attn (64) / _bshd (256) calls

Latency comes from ``triton.testing.do_bench`` (L2 flushed between runs, median).
Peak memory is measured on one call, separately from timing, with the operands
allocated beforehand, so the columns show the combine's own allocations only.
"""
from __future__ import annotations

import argparse
from typing import Callable

import torch
from triton.testing import do_bench

import fastvideo_kernel.ops as ops
from fastvideo_kernel.ops import _combine_coarse_sparse

BE = 64
MIB = 2**20


def _old_combine(out_c, out_s, weight, be, seq_dim):
    """The pre-change combines, transcribed from the old ``video_sparse_attn`` / ``_bshd`` bodies."""
    if seq_dim == 2:  # BHSD: repeat the coarse output to the full sequence, then out-of-place mul and add
        batch, heads, n_blocks, dim = out_c.shape
        out_c = out_c.unsqueeze(3).repeat(1, 1, 1, be, 1).view(batch, heads, n_blocks * be, dim)
        if weight is not None:
            return out_c * weight + out_s
        return out_c + out_s
    batch, n_blocks, heads, dim = out_c.shape  # BSHD: broadcast multiply, out-of-place add
    out_view = out_s.view(batch, n_blocks, be, heads, dim)
    if weight is not None:
        out = out_view + out_c.unsqueeze(2) * weight.view(batch, n_blocks, be, heads, dim)
    else:
        out = out_view + out_c.unsqueeze(2)
    return out.view(batch, n_blocks * be, heads, dim)


def _peak_mib(fn: Callable[[], torch.Tensor]) -> float:
    fn()  # warm up (Triton compilation and autotuning allocate scratch on the first call)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / MIB


def _latency_us(fn: Callable[[], torch.Tensor]) -> float:
    return do_bench(fn, warmup=25, rep=200, return_mode="median") * 1e3


def _row(label: str, fn: Callable[[], torch.Tensor], baseline_us: float | None = None) -> float:
    peak = _peak_mib(fn)
    us = _latency_us(fn)
    speedup = f"{baseline_us / us:>7.2f}x" if baseline_us else f"{'':>8}"
    print(f"  {label:<38}{us:>12.1f}{speedup}{peak:>16.1f}")
    return us


def _header():
    print(f"  {'':<38}{'latency us':>12}{'vs before':>8}{'peak alloc MiB':>16}")


def bench_combine(name, batch, heads, n_blocks, dim, gated):
    seq = n_blocks * BE
    torch.manual_seed(0)
    out_c = torch.randn(batch, heads, n_blocks, dim, device="cuda", dtype=torch.bfloat16)
    out_s = torch.randn(batch, heads, seq, dim, device="cuda", dtype=torch.bfloat16)
    weight = torch.rand(batch, heads, seq, dim, device="cuda", dtype=torch.bfloat16) if gated else None
    # The model's gate is BSHD; the BHSD caller used to copy it into BHSD layout.
    weight_view = (torch.rand(batch, seq, heads, dim, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
                   if gated else None)
    scratch = out_s.clone()  # accumulation target for the in-place variants (values drift; latency does not)
    leaf = out_s.clone().requires_grad_(True)

    print(f"\n{name}  B={batch} H={heads} S={seq} D={dim} gated={gated}  "
          f"(one full [B,H,S,D] bf16 tensor = {out_s.numel() * 2 / MIB:.1f} MiB)")
    _header()
    with torch.no_grad():
        before = _row("before (repeat, mul, add)", lambda: _old_combine(out_c, out_s, weight, BE, 2))
        _row("after, no grad (in place)", lambda: _combine_coarse_sparse(out_c, scratch, weight, BE, 2), before)
        if gated:
            _row("after, no grad, BSHD gate view", lambda: _combine_coarse_sparse(out_c, scratch, weight_view, BE, 2),
                 before)
    with torch.enable_grad():
        _row("after, grad (out of place addcmul)", lambda: _combine_coarse_sparse(out_c, leaf, weight, BE, 2), before)


def bench_e2e(name, heads, n_blocks, dim, gated, ratio=0.2):
    seq = n_blocks * BE
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, heads, seq, dim, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    vbs = torch.full((n_blocks,), BE, device="cuda", dtype=torch.int32)
    gate = torch.rand(1, seq, heads, dim, device="cuda", dtype=torch.bfloat16).transpose(1, 2) if gated else None
    gate_contig = gate.contiguous() if gated else None
    topk = max(1, int(n_blocks * ratio))

    def call(g):
        return ops.video_sparse_attn(q, k, v, vbs, vbs, topk, (4, 4, 4), compress_attn_weight=g)

    print(f"\n{name}  video_sparse_attn B=1 H={heads} S={seq} D={dim} topk={topk}/{n_blocks} gated={gated}  "
          f"(one full tensor = {q.numel() * 2 / MIB:.1f} MiB)")
    _header()
    current = ops._combine_coarse_sparse
    with torch.no_grad():
        ops._combine_coarse_sparse = _old_combine
        try:
            before = _row("before (old combine)", lambda: call(gate_contig))
        finally:
            ops._combine_coarse_sparse = current
        _row("after", lambda: call(gate_contig), before)
        if gated:
            _row("after, BSHD gate view (no copy)", lambda: call(gate), before)


def bench_e2e_bshd(name, heads, n_blocks, dim, gated, ratio=0.2):
    """``video_sparse_attn_bshd`` at 256-token tiles (the production 256-tile route)."""
    be = 256
    n_blocks = (n_blocks * BE) // be
    seq = n_blocks * be
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, seq, heads, dim, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    vbs = torch.full((n_blocks,), be, device="cuda", dtype=torch.int32)
    gate = torch.rand(1, seq, heads, dim, device="cuda", dtype=torch.bfloat16) if gated else None
    topk = max(1, int(n_blocks * ratio))

    def call():
        return ops.video_sparse_attn_bshd(q, k, v, vbs, vbs, topk, (16, 4, 4), compress_attn_weight=gate)

    print(f"\n{name}  video_sparse_attn_bshd B=1 H={heads} S={seq} D={dim} topk={topk}/{n_blocks} gated={gated}  "
          f"(one full tensor = {q.numel() * 2 / MIB:.1f} MiB)")
    _header()
    current = ops._combine_coarse_sparse
    with torch.no_grad():
        ops._combine_coarse_sparse = _old_combine
        try:
            before = _row("before (old combine)", call)
        finally:
            ops._combine_coarse_sparse = current
        _row("after", call, before)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--large", action="store_true", help="add 56x15488x128 and a Wan-14B 480p-like 40x39936x128")
    ap.add_argument("--e2e", action="store_true",
                    help="benchmark full video_sparse_attn (64-token tiles) and video_sparse_attn_bshd (256) calls")
    ap.add_argument("--no-combine", action="store_true", help="skip the combine-only microbenchmark")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm device required")
    print("device:", torch.cuda.get_device_name(0))

    cases = [("small", 1, 8, 64, 128)]
    if args.large:
        # 39,936 tokens = 624 tiles of 64: a 21x30x52 token grid (480p, 81 frames, patch 1x2x2) tiled 4x4x4.
        cases += [("large-15k", 1, 56, 242, 128), ("wan14b-480p-like", 1, 40, 624, 128)]
    if not args.no_combine:
        for name, batch, heads, n_blocks, dim in cases:
            for gated in (False, True):
                bench_combine(name, batch, heads, n_blocks, dim, gated)
    if args.e2e:
        from fastvideo_kernel.block_sparse_attn_256 import _resolve_backend
        print(f"\n128/256-tile sparse backend: {_resolve_backend()} (FASTVIDEO_VSA_CUTEDSL=1 selects the FA4 CuTe route)")
        for name, batch, heads, n_blocks, dim in cases:
            for gated in (False, True):
                bench_e2e(name, heads, n_blocks, dim, gated)
        for name, batch, heads, n_blocks, dim in cases:
            for gated in (False, True):
                bench_e2e_bshd(name, heads, n_blocks, dim, gated)


if __name__ == "__main__":
    main()
