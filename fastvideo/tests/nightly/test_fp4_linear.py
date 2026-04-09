#!/usr/bin/env python3
import math
import sys
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.layers.fp4linear import fp4_linear_forward
from fastvideo.layers.linear import ReplicatedLinear

def fail(msg: str):
    print(f"[FAIL] {msg}")
    sys.exit(1)

def metric_report(ref, test, name, cos_min, rel_l2_max):
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

def run_case(case, fwd_cos_min=0.98, fwd_rel_l2_max=0.15,
             bwd_cos_min=0.999, bwd_rel_l2_max=5e-3):
    device = torch.device("cuda")
    torch.manual_seed(0)

    B = case["batch_size"]
    S = case["seq_len"]
    IN = case["in_features"]
    OUT = case["out_features"]
    NAME = case["name"]
    M = B * S

    print(f"\n== Case: {NAME} | M={M}, K={IN}, N={OUT} ==")

    # Reference BF16 Linear
    ref = nn.Linear(IN, OUT, bias=True).to(device=device, dtype=torch.bfloat16)

    # FP4-forward path is installed by monkey-patching ReplicatedLinear.forward.
    test_layer = ReplicatedLinear(
        IN, OUT, bias=True, params_dtype=torch.float32
    ).to(device)
    test_layer.forward = types.MethodType(fp4_linear_forward, test_layer)

    # Initialize identically from the same fp32 master params
    with torch.no_grad():
        w32 = torch.empty(OUT, IN, device=device, dtype=torch.float32)
        nn.init.kaiming_uniform_(w32, a=math.sqrt(5.0))
        b32 = torch.empty(OUT, device=device, dtype=torch.float32)
        nn.init.uniform_(b32, -1 / math.sqrt(IN), 1 / math.sqrt(IN))

        test_layer.weight.copy_(w32)
        if test_layer.bias is not None:
            test_layer.bias.copy_(b32)

        ref.weight.copy_(w32.to(dtype=torch.bfloat16))
        if ref.bias is not None:
            ref.bias.copy_(b32.to(dtype=torch.bfloat16))

    # Inputs
    x = torch.randn(M, IN, device=device, dtype=torch.bfloat16, requires_grad=True)
    x_ref = x.clone().detach().requires_grad_(True)

    # -------- Forward comparison --------
    with torch.inference_mode():
        y_ref = ref(x_ref)
        y_fp4, _ = test_layer(x)

    print("Forward metrics (FP4-forward vs BF16):")
    metric_report(y_ref, y_fp4, "forward", fwd_cos_min, fwd_rel_l2_max)

    # -------- Backward comparison --------
    # Inject the SAME external grad_out so gradients are comparable.
    grad_out = torch.randn_like(y_ref)

    # Clear grads
    for p in ref.parameters():
        if p.grad is not None:
            p.grad.zero_()
    for p in test_layer.parameters():
        if p.grad is not None:
            p.grad.zero_()
    if x.grad is not None:
        x.grad.zero_()
    if x_ref.grad is not None:
        x_ref.grad.zero_()

    # Recompute forward with grad graph (no inference_mode)
    y_ref = ref(x_ref)
    y_fp4, _ = test_layer(x)

    # Backward with same grad_out
    y_ref.backward(grad_out)
    y_fp4.backward(grad_out)

    # Collect grads (cast to fp32 for fair comparison)
    gx_ref  = x_ref.grad.float()
    gw_ref  = ref.weight.grad.float()
    gb_ref  = ref.bias.grad.float() if ref.bias is not None else None

    gx_fp4  = x.grad.float()
    # test_layer stores weight in fp32 → grad is fp32 already
    gw_fp4  = test_layer.weight.grad.float()
    gb_fp4  = test_layer.bias.grad.float() if test_layer.bias is not None else None

    print("Backward metrics (grads parity with injected grad_out):")
    metric_report(gx_ref, gx_fp4, "grad_x", bwd_cos_min, bwd_rel_l2_max)
    metric_report(gw_ref, gw_fp4, "grad_w", bwd_cos_min, bwd_rel_l2_max)
    if gb_ref is not None and gb_fp4 is not None:
        metric_report(gb_ref, gb_fp4, "grad_b", bwd_cos_min, bwd_rel_l2_max)

def main():
    if not torch.cuda.is_available():
        print("CUDA is required.")
        sys.exit(1)

    # Your exact config list
    configs = [
        {'batch_size': 1, 'seq_len': 1, 'in_features': 256,  'out_features': 5120, 'name': 'fc_in_256x5120'},
        {'batch_size': 1, 'seq_len': 1, 'in_features': 5120, 'out_features': 5120, 'name': 'fc_out_5120x5120'},
        {'batch_size': 1, 'seq_len': 1, 'in_features': 5120, 'out_features': 30720,'name': 'fc_5120x30720'},
        {'batch_size': 1, 'seq_len': 512,'in_features': 4096, 'out_features': 5120, 'name': 'fc_in_512x4096x5120'},
        {'batch_size': 1, 'seq_len': 512,'in_features': 5120, 'out_features': 5120, 'name': 'fc_out_512x5120x5120'},
        {'batch_size': 1, 'seq_len': 32768,'in_features': 5120,'out_features': 5120, 'name': 'to_q_32768x5120x5120'},
        {'batch_size': 1, 'seq_len': 32768,'in_features': 5120,'out_features': 13824,'name': 'ffn_fc_in_32768x5120x13824'},
        {'batch_size': 1, 'seq_len': 32768,'in_features': 13824,'out_features': 5120, 'name': 'ffn_fc_out_32768x13824x5120'},
    ]

    # Tolerances: forward is approximate (FP4), backward should match closely
    fwd_cos_min = 0.98
    fwd_rel_l2_max = 0.15
    bwd_cos_min = 0.999
    bwd_rel_l2_max = 5e-3

    for case in configs:
        run_case(case, fwd_cos_min, fwd_rel_l2_max, bwd_cos_min, bwd_rel_l2_max)
        torch.cuda.synchronize()

    print("\n[OK] All forward & backward checks passed for all cases.")

if __name__ == "__main__":
    main()
