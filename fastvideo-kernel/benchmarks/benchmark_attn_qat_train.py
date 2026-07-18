#!/usr/bin/env python3
"""Benchmark the Attn-QAT training kernel on a single GPU.

Defaults model one rank of the 4-GPU Wan2.1-T2V-1.3B MixKit recipe:
``B=1, H=12/4, L=20*30*52, D=128``.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import torch

from fastvideo_kernel.triton_kernels.attn_qat_train import attention


RTX_5090_DENSE_BF16_TFLOPS = 209.5


def _qat_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    consumer_blackwell = torch.cuda.get_device_capability()[0] == 12
    return attention(
        q,
        k,
        v,
        False,
        q.shape[-1]**-0.5,
        True,  # use_qat_qkv_backward
        False,  # smooth_k
        not consumer_blackwell,  # warp_specialize
        True,  # IS_QAT
        False,  # two_level_quant_P
        True,  # fake_quant_P
        True,  # use_high_prec_o
        False,  # smooth_q
        False,  # use_global_sf_P
        False,  # use_global_sf_QKV
    )


def _measure_ms(fn: Callable[[], object], warmup: int, repeat: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), min(samples), max(samples)


def _format_result(
    label: str,
    timing_ms: tuple[float, float, float],
    algorithmic_flops: int,
    executed_matmul_flops: int,
    peak_tflops: float,
) -> str:
    median_ms, min_ms, max_ms = timing_ms
    algorithmic_tflops = algorithmic_flops / (median_ms * 1e9)
    executed_tflops = executed_matmul_flops / (median_ms * 1e9)
    return (
        f"{label}: {median_ms:.3f} ms (min={min_ms:.3f}, max={max_ms:.3f}), "
        f"algorithmic={algorithmic_tflops:.2f} TFLOPS/{100 * algorithmic_tflops / peak_tflops:.2f}% MFU, "
        f"executed_matmul={executed_tflops:.2f} TFLOPS/{100 * executed_tflops / peak_tflops:.2f}% MFU"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=3, help="Heads per SP rank; Wan 1.3B has 12 total.")
    parser.add_argument("--query-length", type=int, default=31_200)
    parser.add_argument("--kv-length", type=int, help="Defaults to --query-length.")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument(
        "--peak-tflops",
        type=float,
        default=RTX_5090_DENSE_BF16_TFLOPS,
        help="Dense BF16 Tensor TFLOPS with FP32 accumulation; default is RTX 5090 boost-clock peak.",
    )
    args = parser.parse_args()
    kv_length = args.kv_length or args.query_length

    torch.manual_seed(0)
    q_shape = (args.batch_size, args.heads, args.query_length, args.head_dim)
    kv_shape = (args.batch_size, args.heads, kv_length, args.head_dim)
    q = torch.randn(q_shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad_out = torch.randn_like(q)

    compile_start = time.perf_counter()
    output = _qat_attention(q, k, v)
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - compile_start

    forward_ms = _measure_ms(lambda: _qat_attention(q, k, v), args.warmup, args.repeat)
    backward_ms = _measure_ms(
        lambda: torch.autograd.grad(output, (q, k, v), grad_out, retain_graph=True),
        args.warmup,
        args.repeat,
    )

    base_flops = args.batch_size * args.heads * args.query_length * kv_length * args.head_dim
    # Conventional attention FLOPs are 4*base forward and 10*base backward.
    # QAT additionally computes the STE high-precision P@V path in forward and
    # the quantized-P dV path in backward, for 6*base and 14*base matmul FLOPs.
    print(f"device: {torch.cuda.get_device_name()}")
    print(f"q: {q_shape}; k/v: {kv_shape}; compile+first-forward: {compile_seconds:.3f} s")
    print(_format_result("forward", forward_ms, 4 * base_flops, 6 * base_flops, args.peak_tflops))
    print(_format_result("backward", backward_ms, 10 * base_flops, 14 * base_flops, args.peak_tflops))


if __name__ == "__main__":
    main()
