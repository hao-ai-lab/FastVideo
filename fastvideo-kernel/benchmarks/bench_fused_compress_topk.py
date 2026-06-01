#!/usr/bin/env python3
"""
Benchmark fused compress (block mean) and topk mask kernels vs. PyTorch baselines.

Compares:
  compress:
    - Old: .view() -> .float() -> .sum(dim=3) -> / vbs -> .to(bf16)
    - New: fused_block_mean (Triton: bf16 read -> fp32 accumulate -> div -> bf16 write)

  topk:
    - Old: torch.topk() -> zeros() -> scatter_()
    - New: fused_topk_mask (Triton: binary-search pivot -> bool mask)

Reports per-kernel latency (ms), speedup, and numerical accuracy (max abs error,
cosine similarity for compress; mask match rate for topk).
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

try:
    from triton.testing import do_bench
except ImportError as e:
    raise ImportError("This benchmark requires triton (for triton.testing.do_bench).") from e

from fastvideo_kernel.triton_kernels.fused_compress_topk import fused_block_mean, fused_topk_mask


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark fused compress & topk vs. PyTorch baselines")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_heads", type=int, default=12)
    p.add_argument("--head_dim", type=int, default=128, choices=[64, 128])
    p.add_argument("--seq_lens", type=int, nargs="+", default=[49152],
                   help="Sequence lengths to benchmark (must be divisible by block_elements)")
    p.add_argument("--block_elements", type=int, default=64, choices=[64, 256],
                   help="Tokens per block (64 or 256)")
    p.add_argument("--topk", type=int, default=None,
                   help="KV blocks to select per Q block (default: ~10%% of kv_blocks)")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--rep", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Old PyTorch baselines (extracted from ops.py before commit d9cd0b9)
# ---------------------------------------------------------------------------

def pytorch_block_mean(
    x: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    block_elements: int,
) -> torch.Tensor:
    """Old PyTorch compress: .view() -> .float() -> .sum(dim=3) -> / vbs -> .to(dtype)"""
    B, H, seq_len, D = x.shape
    num_blocks = seq_len // block_elements
    x_c = x.view(B, H, num_blocks, block_elements, D)
    x_c = (x_c.float().sum(dim=3) / variable_block_sizes.view(1, 1, -1, 1)).to(x.dtype)
    return x_c


def pytorch_topk_mask(
    scores: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Old PyTorch topk: torch.topk() -> zeros() -> scatter_()"""
    topk = min(topk, scores.shape[-1])
    topk_idx = torch.topk(scores, topk, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, topk_idx, True)
    return mask


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

def accuracy_compress(ref: torch.Tensor, test: torch.Tensor) -> dict:
    ref_f = ref.float()
    test_f = test.float()
    abs_err = (ref_f - test_f).abs()
    cos_sim = torch.nn.functional.cosine_similarity(ref_f.flatten(), test_f.flatten(), dim=0)
    return {
        "max_abs_err": abs_err.max().item(),
        "mean_abs_err": abs_err.mean().item(),
        "cosine_sim": cos_sim.item(),
    }


def accuracy_topk(ref_mask: torch.Tensor, test_mask: torch.Tensor) -> dict:
    match = (ref_mask == test_mask).all(dim=-1).float().mean().item()
    true_per_row_ref = ref_mask.sum(dim=-1).float()
    true_per_row_test = test_mask.sum(dim=-1).float()
    count_match = (true_per_row_ref == true_per_row_test).float().mean().item()
    return {
        "row_exact_match": match,
        "count_match": count_match,
    }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def bench_compress(
    B: int, H: int, seq_len: int, D: int, block_elements: int,
    dtype: torch.dtype, warmup: int, rep: int,
) -> None:
    num_blocks = seq_len // block_elements
    x = torch.randn(B, H, seq_len, D, dtype=dtype, device="cuda")
    vbs = torch.full((num_blocks,), block_elements, dtype=torch.int32, device="cuda")
    # Make a few blocks partially filled to exercise variable block sizes
    if num_blocks > 4:
        vbs[1] = block_elements - 2
        vbs[-2] = block_elements - 5

    # Accuracy
    ref = pytorch_block_mean(x, vbs, block_elements)
    fused = fused_block_mean(x, vbs, block_elements)
    acc = accuracy_compress(ref, fused)

    # Latency
    old_ms = do_bench(lambda: pytorch_block_mean(x, vbs, block_elements), warmup=warmup, rep=rep)
    new_ms = do_bench(lambda: fused_block_mean(x, vbs, block_elements), warmup=warmup, rep=rep)

    speedup = old_ms / new_ms if new_ms > 0 else float("inf")
    print(f"  compress  | old: {old_ms:8.3f} ms | new: {new_ms:8.3f} ms | speedup: {speedup:5.2f}x "
          f"| max_abs_err: {acc['max_abs_err']:.2e} | cos_sim: {acc['cosine_sim']:.8f}")


def bench_topk(
    B: int, H: int, num_blocks: int, topk: int,
    dtype: torch.dtype, warmup: int, rep: int,
) -> None:
    scores = torch.randn(B, H, num_blocks, num_blocks, dtype=dtype, device="cuda")

    # Accuracy
    ref = pytorch_topk_mask(scores, topk)
    fused = fused_topk_mask(scores, topk)
    acc = accuracy_topk(ref, fused)

    # Latency
    old_ms = do_bench(lambda: pytorch_topk_mask(scores, topk), warmup=warmup, rep=rep)
    new_ms = do_bench(lambda: fused_topk_mask(scores, topk), warmup=warmup, rep=rep)

    speedup = old_ms / new_ms if new_ms > 0 else float("inf")
    print(f"  topk      | old: {old_ms:8.3f} ms | new: {new_ms:8.3f} ms | speedup: {speedup:5.2f}x "
          f"| row_exact_match: {acc['row_exact_match']:.4f} | count_match: {acc['count_match']:.4f}")


def main() -> None:
    args = parse_arguments()
    set_seed(args.seed)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    B, H, D = args.batch_size, args.num_heads, args.head_dim
    block_elements = args.block_elements

    print("Fused Compress & TopK Benchmark")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"batch={B}, heads={H}, head_dim={D}, block_elements={block_elements}, dtype={args.dtype}")
    print(f"warmup={args.warmup}, rep={args.rep}")

    for seq_len in args.seq_lens:
        if seq_len % block_elements != 0:
            print(f"\n[skip] seq_len={seq_len} not divisible by block_elements={block_elements}")
            continue

        num_blocks = seq_len // block_elements
        topk = args.topk if args.topk is not None else max(1, num_blocks // 10)
        topk = min(topk, num_blocks)

        print(f"\n{'=' * 100}")
        print(f"seq_len={seq_len}, num_blocks={num_blocks}, topk={topk}")
        print("-" * 100)

        bench_compress(B, H, seq_len, D, block_elements, dtype, args.warmup, args.rep)
        bench_topk(B, H, num_blocks, topk, dtype, args.warmup, args.rep)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    main()
