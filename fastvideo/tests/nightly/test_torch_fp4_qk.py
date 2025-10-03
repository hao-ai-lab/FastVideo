#!/usr/bin/env python3
import math
import sys
import torch
import torch.nn.functional as F
from fastvideo.attention.backends.sage_attn3 import fp_16_qk, sage_attn_qk_torch


def fail(msg: str):
    print(f"[FAIL] {msg}")
    sys.exit(1)


def metric_report(ref, test, name, cos_min=None, rel_l2_max=None):
    diff = (test - ref).float()
    ref32 = ref.float()
    test32 = test.float()

    mean_abs_diff = diff.abs().mean().item()
    max_abs_diff  = diff.abs().max().item()
    rel_l2 = (diff.norm() / (ref32.norm() + 1e-12)).item()
    cos = F.cosine_similarity(ref32.reshape(-1), test32.reshape(-1), dim=0).item()

    print(f"  [{name}] cos: {cos:.6f}, rel_l2: {rel_l2:.6e}, "
          f"mean_abs: {mean_abs_diff:.6e}, max_abs: {max_abs_diff:.6e}")

    if cos_min is not None and cos < cos_min:
        fail(f"{name}: cosine similarity too low: {cos:.6f} < {cos_min}")
    if rel_l2_max is not None and rel_l2 > rel_l2_max:
        fail(f"{name}: relative L2 too high: {rel_l2:.6e} > {rel_l2_max}")


def run_case(case, fwd_cos_min=0.98, fwd_rel_l2_max=2.0e-1):
    device = torch.device("cuda")
    torch.manual_seed(0)

    B, H, L, D = case["B"], case["H"], case["L"], case["D"]
    NAME = case["name"]
    assert B == 1, "sage_attn_qk_torch requires B == 1"

    print(f"\n== Case: {NAME} | B={B}, H={H}, L={L}, D={D} ==")

    dtype = torch.bfloat16
    q = torch.randn(B, H, L, D, device=device, dtype=dtype)
    k = torch.randn(B, H, L, D, device=device, dtype=dtype)

    with torch.inference_mode():
        # Reference computes q @ k^T / sqrt(D)
        p_ref = fp_16_qk(q, k.transpose(-1, -2))          # [B,H,L,L]
        p_sage = sage_attn_qk_torch(q, k, False)           # [B,H,L,L]
    print("Forward metrics (sage_attn FP4 vs fp16/bf16 reference):")
    metric_report(p_ref, p_sage, "forward", fwd_cos_min, fwd_rel_l2_max)


def main():
    try:
        _ = fp_16_qk
        _ = sage_attn_qk_torch
    except NameError:
        print("Please import/define fp_16_qk and sage_attn_qk_torch before running.")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("CUDA is required.")
        sys.exit(1)

    configs = [
        {"B": 1, "H": 4,  "L": 256, "D": 128, "name": "H4_L256_D128"},
        {"B": 1, "H": 8,  "L": 512, "D": 128, "name": "H8_L512_D128"},
        {"B": 1, "H": 8,  "L": 256, "D": 256, "name": "H8_L256_D256"},
        {"B": 1, "H": 16, "L": 384, "D": 128, "name": "H16_L384_D128"},
    ]

    fwd_cos_min = 0.98
    fwd_rel_l2_max = 2.0e-1  # relax for FP4 probabilities

    for case in configs:
        run_case(case, fwd_cos_min=fwd_cos_min, fwd_rel_l2_max=fwd_rel_l2_max)
        torch.cuda.synchronize()

    print("\n[OK] All forward diff checks passed for all cases.")


if __name__ == "__main__":
    main()