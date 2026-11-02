# SPDX-License-Identifier: Apache-2.0
"""In-framework benchmark of the ptx_sm107a VSA backend (Rubin, 128-token blocks) at real Wan shapes.

Builds the metadata FastVideo builds, at the deployed latent shapes with the tile-128 tiling (4, 4, 8), and
calls the backend with index tensors made by map_to_index, with raw q and with q pre-scaled by log2(e)/sqrt(d).
No Triton column: the Triton block-sparse forward does not compile for sm_107a with Triton 3.7 (torch 2.12).

    PYTHONPATH=fastvideo-kernel/python python tests/bench_block_sparse_ptx_sm107a.py
"""

import time

import torch

from fastvideo_kernel import block_sparse_attn_ptx_sm107a as vsa
from fastvideo_kernel.triton_kernels.index import map_to_index
from fastvideo_kernel.vsa_utils import build_vsa_metadata

HEAD_DIM = 128

# (label, latent (T,H,W), tile, heads). The latents are Wan's; heads is per-rank.
CASES = [
    ("480P", (21, 30, 52), (4, 4, 8), 40),
    ("720P", (21, 45, 80), (4, 4, 8), 40),
]


def timed(fn, iters=200, warmup=100):   # long enough to reach the power-limited clock; short runs favor the first
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    if torch.cuda.get_device_capability() != (10, 7):
        print("not Rubin (sm_107); skipping")
        return
    print(f"{'case':<8} {'S_pad':>7} {'blocks':>7} {'topk':>5} {'raw q ms':>9} {'TFLOPS':>7} "
          f"{'pre q ms':>9} {'TFLOPS':>7}  selected")
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

        q_pre = (q.float() * (1.4426950408889634 / HEAD_DIM**0.5)).to(torch.bfloat16)

        ok = vsa.is_supported(q, vbs)
        nan = float("nan")
        raw = timed(lambda: vsa.block_sparse_attn_ptx_sm107a(q, k, v, idx, num, vbs)) if ok else nan
        pre = timed(lambda: vsa.block_sparse_attn_ptx_sm107a(q_pre, k, v, idx, num, vbs, q_prescaled=True)) if ok else nan
        flops = 4.0 * heads * S * topk * block * HEAD_DIM   # over the selected blocks

        print(f"{label:<8} {S:>7} {nb:>7} {topk:>5} {raw:>9.3f} {flops / raw / 1e9:>7.1f} "
              f"{pre:>9.3f} {flops / pre / 1e9:>7.1f}  {'sm107a' if ok else 'unsupported'}")


if __name__ == "__main__":
    main()
