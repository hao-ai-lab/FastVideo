#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import sys
import math
import torch
import torch.nn.functional as F

# ---- Import function-under-test ----
# Make sure this import path matches your project layout.
# If it's in the same file, remove this import and keep the definition above.
from fastvideo.attention.backends.sage_attn3 import sageattn_blackwell_with_16bit_bwd

# ---- Import FlashAttention forward (public) ----
# We DO NOT use _wrapped_* internals. Prefer the top-level alias; fall back if needed.
try:
    from flash_attn import flash_attn_func as flash_attn
except Exception:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func as flash_attn
    except Exception as e:
        print("Could not import flash_attn function:", e)
        sys.exit(1)


def fail(msg: str):
    print(f"[FAIL] {msg}")
    sys.exit(1)


def metric_report(ref, test, name, cos_min=None, rel_l2_max=None):
    ref32 = ref.float()
    test32 = test.float()
    diff = test32 - ref32

    mean_abs = diff.abs().mean().item()
    max_abs  = diff.abs().max().item()
    rel_l2   = (diff.norm() / (ref32.norm() + 1e-12)).item()
    cos      = F.cosine_similarity(ref32.reshape(-1), test32.reshape(-1), dim=0).item()

    print(f"  [{name}] cos={cos:.6f}  rel_l2={rel_l2:.6e}  mean_abs={mean_abs:.6e}  max_abs={max_abs:.6e}")
    if cos_min is not None and cos < cos_min:
        fail(f"{name}: cosine too low: {cos:.6f} < {cos_min}")
    if rel_l2_max is not None and rel_l2 > rel_l2_max:
        fail(f"{name}: relative L2 too high: {rel_l2:.6e} > {rel_l2_max}")


def run_case(case, fwd_cos_min=0.95, fwd_rel_l2_max=0.25, bwd_cos_min=0.95, bwd_rel_l2_max=0.25):
    if not torch.cuda.is_available():
        print("CUDA is required.")
        sys.exit(1)

    device = torch.device("cuda")
    torch.manual_seed(case.get("seed", 0))

    B = case["B"]
    L = case["L"]
    H = case["H"]
    D = case["D"]
    dtype = case["dtype"]
    causal = case["causal"]
    name = case["name"]

    print(f"\n== Case: {name} | B={B}, L={L}, H={H}, D={D}, dtype={str(dtype).split('.')[-1]}, causal={causal} ==")

    # Inputs in BLHD (flash_attn expects BLHD)
    q = torch.randn(B, L, H, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, L, H, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, L, H, D, device=device, dtype=dtype, requires_grad=True)

    softmax_scale = 1.0 / math.sqrt(D)

    # ----- Forward -----
    # Our kernel (forward uses sage, backward will use FA inside the autograd fn)
    y_sage = sageattn_blackwell_with_16bit_bwd(q, k, v, is_causal=causal)

    # Sanity checks
    if y_sage.shape != (B, L, H, D):
        fail(f"Output shape mismatch: got {tuple(y_sage.shape)}, expected {(B, L, H, D)}")
    if not torch.isfinite(y_sage).all():
        fail("Non-finite values detected in forward output")

    # FlashAttention reference forward (public function)
    with torch.inference_mode():
        y_fa = flash_attn(q, k, v, dropout_p=0.0, softmax_scale=softmax_scale, causal=causal)

    print("Forward numeric check vs FlashAttention:")
    metric_report(y_fa, y_sage, "forward", cos_min=fwd_cos_min, rel_l2_max=fwd_rel_l2_max)

    # ----- Backward -----
    # Inject the SAME grad_out for both models so gradients are comparable
    grad_out = torch.randn_like(y_sage)

    # Keep separate copies with grad for the FA path
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    # Backprop through our function
    # (It recomputes FA fwd+bwd internally in 16-bit per your implementation)
    for t in (q, k, v):
        if t.grad is not None:
            t.grad.zero_()
    y_sage.backward(grad_out)
    dq_sage, dk_sage, dv_sage = q.grad.detach(), k.grad.detach(), v.grad.detach()

    # Backprop through FA public forward
    for t in (q_ref, k_ref, v_ref):
        if t.grad is not None:
            t.grad.zero_()
    y_ref = flash_attn(q_ref, k_ref, v_ref, dropout_p=0.0, softmax_scale=softmax_scale, causal=causal)
    y_ref.backward(grad_out)

    dq_ref, dk_ref, dv_ref = q_ref.grad.detach(), k_ref.grad.detach(), v_ref.grad.detach()

    print("Backward gradient checks vs FlashAttention:")
    metric_report(dq_ref, dq_sage, "grad_q", cos_min=bwd_cos_min, rel_l2_max=bwd_rel_l2_max)
    metric_report(dk_ref, dk_sage, "grad_k", cos_min=bwd_cos_min, rel_l2_max=bwd_rel_l2_max)
    metric_report(dv_ref, dv_sage, "grad_v", cos_min=bwd_cos_min, rel_l2_max=bwd_rel_l2_max)

def main():
    # Keep sizes moderate so it runs quickly; adjust as you like.
    cases = [
        {"name": "bf16_noncausal_L16384", "B": 1, "L": 1024, "H": 8,  "D": 128, "dtype": torch.bfloat16, "causal": False},
        {"name": "bf16_noncausal_L2048", "B": 1, "L": 2048, "H": 8,  "D": 128, "dtype": torch.bfloat16, "causal": False},
    ]

    for case in cases:
        run_case(case)
        torch.cuda.synchronize()

    print("\n[OK] All tests passed.")


if __name__ == "__main__":
    main()
