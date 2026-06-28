#!/usr/bin/env python3
"""
Correctness tests for the sageattn_blackwell (ATTN_QAT_INFER) FP4 inference kernel.

Compares causal and non-causal outputs against a naive float32 reference to verify
that the V-row permutation in scaled_fp4_quant_trans_kernel is correct (or absent).

Requires a Blackwell GPU (sm_120a) and fp4attn_cuda / fp4quant_cuda extensions built
via `cd fastvideo-kernel && ./build.sh`.

Run from the fastvideo-kernel directory:
    python tests/test_attn_qat_infer.py
or via pytest:
    pytest tests/test_attn_qat_infer.py -v
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from attn_qat_infer.api import sageattn_blackwell
# sys.path.insert(0, "/home/kevin/workspace/SageAttention/sageattention3_blackwell")
# sys.path.insert(0, "/home/kevin/workspace/SageAttentionE/sageattention3_blackwell")
# from sageattn3.api import sageattn3_blackwell as sageattn_blackwell

DEVICE = torch.device("cuda")


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    norm_a = torch.norm(a_flat)
    norm_b = torch.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return (torch.dot(a_flat, b_flat) / (norm_a * norm_b)).item()


def reference_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   is_causal: bool) -> torch.Tensor:
    """PyTorch math-backend SDPA reference in float32."""
    with sdpa_kernel([SDPBackend.MATH]):
        out = F.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), is_causal=is_causal
        )
    return out.to(q.dtype)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench(B, H, L, D, is_causal, warmup=20, iters=100):
    q = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16)

    for _ in range(warmup):
        sageattn_blackwell(q, k, v, is_causal=is_causal)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        sageattn_blackwell(q, k, v, is_causal=is_causal)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / iters
    flops = 4 * B * H * L * L * D / 1e12
    tflops = flops / (ms / 1e3)
    return ms, tflops


def run_benchmark():
    B, H = 2, 32
    print(f"{'B':>2} {'H':>2} {'L':>5} {'D':>4} {'causal':>6}  {'ms':>8}  {'TFLOPS':>7}")
    print("-" * 50)
    for is_causal in [False, True]:
        for L in [512, 1024, 2048, 4096]:
            for D in [64, 128]:
                ms, tflops = bench(B, H, L, D, is_causal)
                print(
                    f"{B:>2} {H:>2} {L:>5} {D:>4} {str(is_causal):>6}"
                    f"  {ms:>8.3f}  {tflops:>7.2f}"
                )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

import pytest


@pytest.mark.parametrize("B,H,L,D", [
    (2, 4,  128,  64), (2, 4,  128, 128),
    (2, 4,  256,  64), (2, 4,  256, 128),
    (2, 4,  512,  64), (2, 4,  512, 128),
    (2, 4, 1024,  64), (2, 4, 1024, 128),
    (2, 4, 2048,  64), (2, 4, 2048, 128),
    (2, 4, 4096,  64), (2, 4, 4096, 128),
    (1, 4,  128, 128),
    (2, 8,  256, 128),
    (1, 4,  384, 128),
])
@pytest.mark.parametrize("causal", [False, True], ids=["non_causal", "causal"])
def test_accuracy_sdpa(causal: bool, B: int, H: int, L: int, D: int):
    """SDPA-reference accuracy sweep with 0.3-scaled inputs."""
    torch.manual_seed(42)
    q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device=DEVICE)
    k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device=DEVICE)
    v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device=DEVICE)
    ref = reference_sdpa(q, k, v, is_causal=causal)
    out = sageattn_blackwell(q.clone(), k.clone(), v.clone(), is_causal=causal)
    cos = cosine_similarity(out, ref)
    label = "causal" if causal else "non-causal"
    print(f"  [{label}] B={B} H={H} L={L} D={D} — cos_sim={cos:.6f}")
    assert cos >= 0.97, f"({label}) B={B} H={H} L={L} D={D} cos_sim={cos:.4f} < 0.97"


if __name__ == "__main__":
    print("=" * 65)
    print("sageattn_blackwell (ATTN_QAT_INFER) inference correctness tests")
    print("=" * 65)
    print()

    print("Accuracy sweep — SDPA reference:")
    for causal in [False, True]:
        for B, H, L, D in [
            (2, 4,  128,  64), (2, 4,  128, 128),
            (2, 4,  256,  64), (2, 4,  256, 128),
            (2, 4,  512,  64), (2, 4,  512, 128),
            (2, 4, 1024,  64), (2, 4, 1024, 128),
            (2, 4, 2048,  64), (2, 4, 2048, 128),
            (2, 4, 4096,  64), (2, 4, 4096, 128),
            (1, 4,  128, 128),
            (2, 8,  256, 128),
            (1, 4,  384, 128),
        ]:
            test_accuracy_sdpa(causal, B, H, L, D)
    print()
    print("All tests completed.")
    print()
    print("Benchmark (B=2, H=32, bf16):")
    run_benchmark()
