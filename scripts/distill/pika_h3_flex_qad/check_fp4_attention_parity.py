# SPDX-License-Identifier: Apache-2.0
"""Parity of the dense FP4 attention emulator (QAD training) with FlashAttention-4 NVFP4 (deployment).

With ``--qk-transform`` both sides precondition Q/K first (the emulator through ``qk_transform=``, the deployed
``FlashAttentionImpl._forward_nvfp4`` through the ``nvfp4_qk_transform`` impl arg). Both should compute Q/K in NVFP4
(per-16 E4M3 block scales, unit global scale) with P and V in BF16. Reports the relative L2 distance of the emulator
to the FA4 NVFP4 kernel next to the FP4-vs-BF16 distance: the emulator is a faithful training target when the first
is a small fraction of the second. Needs a Blackwell GPU with FA4 FP4.

    python scripts/distill/pika_h3_flex_qad/check_fp4_attention_parity.py --seq-len 8269 --heads 14 \\
        --qk-transform smooth_k+hadamard
"""
from __future__ import annotations

import argparse

import torch


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq-len", type=int, default=8269, help="not a multiple of 128 on purpose")
    parser.add_argument("--heads", type=int, default=14)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--qk-transform", default="none", help="none, smooth_k, hadamard or smooth_k+hadamard")
    parser.add_argument("--k-bias", type=float, default=4.0, help="per-channel K offset that smooth_k removes")
    parser.add_argument("--outlier-scale", type=float, default=8.0, help="scale of a few outlier K channels")
    args = parser.parse_args()

    from fastvideo.attention.backends import flash_attn as fa
    from fastvideo.attention.backends.fp4_dense_qat import FP4DenseAttention
    from fastvideo.attention.backends.fp4_vsa_qat import FP4AttentionNumerics
    from fastvideo.attention.utils.nvfp4_qk_precondition import parse_qk_transform

    torch.manual_seed(0)
    shape = (1, args.seq_len, args.heads, args.dim)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    k[..., :4] *= args.outlier_scale
    k += args.k_bias * torch.randn(1, 1, args.heads, args.dim, device="cuda", dtype=torch.bfloat16)
    transform = parse_qk_transform(args.qk_transform)
    scale = args.dim**-0.5

    impl = fa.FlashAttentionImpl(num_heads=args.heads, head_size=args.dim, causal=False, softmax_scale=scale,
                                 num_kv_heads=args.heads, nvfp4_fa4=True, nvfp4_qk_transform=args.qk_transform)
    fa4_fp4 = impl._forward_nvfp4(q, k, v)
    bf16 = torch.nn.functional.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (q, k, v)),
                                                           scale=scale).transpose(1, 2)
    numerics = FP4AttentionNumerics(quantize=True, quantize_pv=False)
    emulator = FP4DenseAttention(numerics, qk_transform=transform)(q, k, v, None)

    print(f"qk_transform={args.qk_transform} k_bias={args.k_bias} outlier_scale={args.outlier_scale}")
    fp4_vs_bf16 = rel(fa4_fp4, bf16)
    emu_vs_fa4 = rel(emulator, fa4_fp4)
    print(f"FA4 NVFP4 vs BF16 attention: rel L2 {fp4_vs_bf16:.5f}")
    print(f"emulator  vs BF16 attention: rel L2 {rel(emulator, bf16):.5f}")
    print(f"emulator  vs FA4 NVFP4:      rel L2 {emu_vs_fa4:.5f}  max |diff| "
          f"{float((emulator.float() - fa4_fp4.float()).abs().max()):.4e}")
    ok = emu_vs_fa4 <= 0.1 * fp4_vs_bf16
    print(f"PARITY {'OK' if ok else 'MISMATCH'}: emulator-to-kernel distance is "
          f"{emu_vs_fa4 / max(fp4_vs_bf16, 1e-12):.1%} of the FP4 quantization error")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
