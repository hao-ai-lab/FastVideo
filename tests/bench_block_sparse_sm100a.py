# SPDX-License-Identifier: Apache-2.0
"""In-framework benchmark of the sm_100a VSA backend at real Wan shapes.

Not the powers-of-two bench grid: this builds the metadata FastVideo actually builds, at the
deployed latent shapes, and drives the same block_sparse_attn_from_indices entry point the
model calls -- so it measures the path that will run, including the dispatch and the index
tensors as constructed rather than synthesised.

    PYTHONPATH=fastvideo-kernel/python python tests/bench_block_sparse_sm100a.py
"""

import time

import torch

from fastvideo_kernel import block_sparse_attn_sm100a as vsa
from fastvideo_kernel.block_sparse_attn import block_sparse_attn_triton
from fastvideo_kernel.triton_kernels.index import map_to_index
from fastvideo_kernel.vsa_utils import build_vsa_metadata

HEAD_DIM = 128

# (label, latent (T,H,W), tile, heads). The latents are Wan's; heads is per-rank.
CASES = [
    ("480P", (21, 30, 52), (4, 4, 4), 40),
    ("720P", (21, 45, 80), (4, 4, 4), 40),
]


def timed(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    if torch.cuda.get_device_capability() != (10, 0):
        print("not Blackwell; skipping")
        return
    print(f"{'case':<8} {'S_pad':>7} {'blocks':>7} {'topk':>5} {'sm100a ms':>10} "
          f"{'triton ms':>10} {'speedup':>8}  selected")
    for label, latent, tile, heads in CASES:
        meta = build_vsa_metadata(latent, tile_size=tile, device="cuda")
        vbs = meta["variable_block_sizes"].to(torch.int32)
        nb = vbs.numel()
        block = int(meta["max_block_size"])
        S = nb * block
        topk = max(1, int(0.1 * nb))          # sparsity 0.9, as deployed

        q, k, v = (torch.randn(1, heads, S, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
                   for _ in range(3))
        # index tensors exactly as FastVideo builds them, from a top-k bool map
        scores = torch.randn(1, heads, nb, nb, device="cuda")
        keep = torch.zeros_like(scores, dtype=torch.bool)
        keep.scatter_(-1, scores.topk(topk, dim=-1).indices, True)
        idx, num = map_to_index(keep)
        idx, num = idx.to(torch.int32).contiguous(), num.to(torch.int32).contiguous()

        ok = vsa.is_supported(q, vbs)
        ours = timed(lambda: vsa.block_sparse_attn_sm100a(q, k, v, idx, num, vbs)) if ok else float("nan")

        # Triton needs 64-token blocks: expand each 128-block into its two halves.
        if block == 128:   # Triton is 64-granular; expand only when we run 128
            keep64 = keep.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
            vbs64 = torch.stack([vbs.clamp(max=64), (vbs - 64).clamp(min=0)], dim=-1).flatten()
            i64, n64 = map_to_index(keep64)
            i64, n64 = i64.to(torch.int32).contiguous(), n64.to(torch.int32).contiguous()
            vbs64 = vbs64.to(torch.int32).contiguous()
        else:
            keep64, i64, n64, vbs64 = keep, idx, num, vbs
        tri = timed(lambda: block_sparse_attn_triton(q, k, v, i64, n64, vbs64))

        print(f"{label:<8} {S:>7} {nb:>7} {topk:>5} {ours:>10.3f} {tri:>10.3f} "
              f"{tri / ours:>7.2f}x  {'sm100a' if ok else 'FALLBACK'}")


if __name__ == "__main__":
    main()
