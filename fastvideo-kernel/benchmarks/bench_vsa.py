#!/usr/bin/env python3
"""
Benchmark VSA block-sparse attention kernels (forward + backward) and report TFLOPs.

This script benchmarks the *raw CUDA kernels* exposed by the extension:
  - fastvideo_kernel_ops.block_sparse_fwd
  - fastvideo_kernel_ops.block_sparse_bwd

It generates a random block-sparse pattern and converts it to index format on GPU.
"""

from __future__ import annotations

import argparse
import math
import random
from typing import Tuple

import numpy as np
import torch

try:
    from triton.testing import do_bench
except Exception as e:  # pragma: no cover
    raise ImportError("This benchmark requires triton (for triton.testing.do_bench).") from e


BLOCK_M = 64
BLOCK_N = 64


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark FastVideo VSA block-sparse attention")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_heads", type=int, default=12)
    p.add_argument("--head_dim", type=int, default=128, choices=[64, 128])
    p.add_argument("--topk", type=int, default=None, help="KV blocks per Q block (default: ~90%% sparsity)")
    p.add_argument("--q_seq_lens", type=int, nargs="+", default=[49152], help="Q sequence lengths (must be /64)")
    p.add_argument("--kv_seq_lens", type=int, nargs="+", default=None, help="KV sequence lengths (defaults to q_seq_len)")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--rep", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    return p.parse_args()


def create_qkv(batch: int, heads: int, q_len: int, kv_len: int, d: int, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(batch, heads, q_len, d, dtype=dtype, device="cuda")
    k = torch.randn(batch, heads, kv_len, d, dtype=dtype, device="cuda")
    v = torch.randn(batch, heads, kv_len, d, dtype=dtype, device="cuda")
    return q, k, v


def make_block_map(bs: int, h: int, num_q_blocks: int, num_kv_blocks: int, topk: int) -> torch.Tensor:
    # block_map: [bs, h, num_q_blocks, num_kv_blocks] bool
    scores = torch.rand(bs, h, num_q_blocks, num_kv_blocks, device="cuda")
    topk = min(max(1, topk), num_kv_blocks)
    idx = torch.topk(scores, topk, dim=-1).indices
    block_map = torch.zeros(bs, h, num_q_blocks, num_kv_blocks, dtype=torch.bool, device="cuda")
    block_map.scatter_(-1, idx, True)
    return block_map


def map_to_index(block_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # GPU conversion (Triton) – avoids Python loops / nonzero
    from fastvideo_kernel.triton_kernels.index import map_to_index as _map_to_index
    return _map_to_index(block_map)


def flops_sparse_attention(bs: int, h: int, d: int, q_len: int, topk_blocks: int, block_n: int) -> float:
    # Approx: QK^T + PV, each is ~2*bs*h*q_len*(topk_blocks*block_n)*d
    return 4.0 * bs * h * d * q_len * (topk_blocks * block_n)


def main() -> None:
    args = parse_arguments()
    set_seed(args.seed)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    try:
        from fastvideo_kernel._C import fastvideo_kernel_ops
    except Exception as e:
        raise RuntimeError(
            "fastvideo-kernel extension not available. Build/install fastvideo-kernel first."
        ) from e

    block_sparse_fwd = getattr(fastvideo_kernel_ops, "block_sparse_fwd", None)
    block_sparse_bwd = getattr(fastvideo_kernel_ops, "block_sparse_bwd", None)
    if block_sparse_fwd is None or block_sparse_bwd is None:
        raise RuntimeError("block_sparse_fwd/bwd not found in fastvideo_kernel_ops.")

    bs, h, d = args.batch_size, args.num_heads, args.head_dim
    kv_seq_lens = args.kv_seq_lens
    if kv_seq_lens is None:
        kv_seq_lens = args.q_seq_lens
    if len(kv_seq_lens) != len(args.q_seq_lens):
        raise ValueError("kv_seq_lens must have the same number of entries as q_seq_lens (or be omitted).")

    print("VSA Block-Sparse Attention Benchmark")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"batch={bs}, heads={h}, head_dim={d}, dtype={args.dtype}")
    print(f"BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}")

    for q_len, kv_len in zip(args.q_seq_lens, kv_seq_lens):
        if q_len % BLOCK_M != 0 or kv_len % BLOCK_N != 0:
            print(f"[skip] q_len={q_len}, kv_len={kv_len} must be divisible by 64")
            continue

        num_q_blocks = q_len // BLOCK_M
        num_kv_blocks = kv_len // BLOCK_N
        topk = args.topk if args.topk is not None else max(1, num_kv_blocks // 10)
        topk = min(topk, num_kv_blocks)

        print("\n" + "=" * 80)
        print(f"q_len={q_len}, kv_len={kv_len}, num_q_blocks={num_q_blocks}, num_kv_blocks={num_kv_blocks}, topk={topk}")

        q, k, v = create_qkv(bs, h, q_len, kv_len, d, dtype)
        block_map = make_block_map(bs, h, num_q_blocks, num_kv_blocks, topk)

        q2k_index, q2k_num = map_to_index(block_map)
        k2q_index, k2q_num = map_to_index(block_map.transpose(-1, -2).contiguous())

        # Variable block sizes: default full blocks (64 tokens per KV block)
        variable_block_sizes = torch.full((num_kv_blocks,), BLOCK_N, dtype=torch.int32, device="cuda")

        # Warmup + capture outputs needed for bwd
        o, lse = block_sparse_fwd(q, k, v, q2k_index, q2k_num, variable_block_sizes)
        og = torch.randn_like(o)
        torch.cuda.synchronize()

        fwd_ms = do_bench(
            lambda: block_sparse_fwd(q, k, v, q2k_index, q2k_num, variable_block_sizes),
            warmup=args.warmup,
            rep=args.rep,
            quantiles=None,
        )

        # Backward warmup
        for _ in range(args.warmup):
            block_sparse_bwd(q, k, v, o, lse, og, k2q_index, k2q_num, variable_block_sizes)
        torch.cuda.synchronize()

        bwd_ms = do_bench(
            lambda: block_sparse_bwd(q, k, v, o, lse, og, k2q_index, k2q_num, variable_block_sizes),
            warmup=0,
            rep=args.rep,
            quantiles=None,
        )

        flops = flops_sparse_attention(bs, h, d, q_len, topk, BLOCK_N)
        fwd_tflops = flops / fwd_ms * 1e-12 * 1e3
        # Rough backward multiplier (attention backward typically ~2-3x forward)
        bwd_tflops = (2.5 * flops) / bwd_ms * 1e-12 * 1e3

        print(f"fwd: {fwd_ms:.3f} ms  | {fwd_tflops:.2f} TFLOPs (approx)")
        print(f"bwd: {bwd_ms:.3f} ms  | {bwd_tflops:.2f} TFLOPs (approx, 2.5x fwd flops)")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    main()


