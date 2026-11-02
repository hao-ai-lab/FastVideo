# SPDX-License-Identifier: Apache-2.0
"""Compare H3 vision attention batching with synthetic inputs and random weights.

Run from the repository root with PYTHONPATH=. in a prepared FastVideo environment::

    python tests/local_tests/encoders/benchmark_minimax_h3_vision_attention.py \
        --device cuda --dtype bfloat16 --segments 16 --sequence-length 1024

This measures one native vision-attention module, including QKV, RoPE and output
projection. It does not measure checkpoint quality or end-to-end H3 latency.
Batch size 1 is the unchanged singleton SDPA baseline; timings exclude warmup.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from fastvideo import envs
from fastvideo.configs.models.encoders.minimax_h3_qwen3_vl import (
    MiniMaxH3Qwen3VLArchConfig,
    MiniMaxH3Qwen3VLConfig,
)
from fastvideo.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLVisionAttention


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--segments", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=1152)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    positive = [args.segments, args.sequence_length, args.hidden_size, args.heads, args.runs, *args.batch_sizes]
    if any(value < 1 for value in positive) or args.warmup < 0:
        parser.error("dimensions, runs and batch sizes must be positive; warmup must be non-negative")
    if args.hidden_size % args.heads or (args.hidden_size // args.heads) % 2:
        parser.error("hidden size must divide evenly into heads with even head dimensions")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu for a smoke check only")
    if args.device == "cuda" and args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        parser.error("this CUDA device does not support bfloat16")

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(42)
    config = MiniMaxH3Qwen3VLConfig(arch_config=MiniMaxH3Qwen3VLArchConfig(
        vision_hidden_size=args.hidden_size,
        vision_num_heads=args.heads,
    ))
    model = MiniMaxH3Qwen3VLVisionAttention(config).to(device=device, dtype=dtype).eval()
    rows = args.segments * args.sequence_length
    hidden = torch.randn(rows, args.hidden_size, device=device, dtype=dtype)
    angles = torch.randn(rows, args.hidden_size // args.heads // 2, device=device)
    angles = torch.cat((angles, angles), dim=-1)
    positions = (angles.cos(), angles.sin())
    lengths = [args.sequence_length] * args.segments
    batch_sizes = list(dict.fromkeys([1, *args.batch_sizes]))

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def run(batch_size: int) -> torch.Tensor:
        with envs.FASTVIDEO_H3_VISION_ATTN_BATCH_SIZE.override(batch_size):
            return model(hidden, lengths, positions)

    results = []
    with torch.inference_mode():
        reference = run(1)
        for batch_size in batch_sizes:
            for _ in range(args.warmup):
                warm_output = run(batch_size)
                del warm_output
            synchronize()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            timings = []
            for _ in range(args.runs):
                synchronize()
                start = time.perf_counter()
                output = run(batch_size)
                synchronize()
                timings.append((time.perf_counter() - start) * 1000)
                del output
            peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            # Compare outside the timing and memory-measurement interval.
            output = run(batch_size)
            error = (output.float() - reference.float()).abs()
            tolerance = (1e-2, 3e-3) if dtype == torch.bfloat16 else (1e-4, 1e-5)
            torch.testing.assert_close(output, reference, rtol=tolerance[0], atol=tolerance[1])
            results.append({
                "batch_size": batch_size,
                "sdpa_calls_per_forward": (args.segments + batch_size - 1) // batch_size,
                "median_wall_ms": statistics.median(timings),
                "wall_ms": timings,
                "peak_allocated_bytes": peak,
                "max_abs_error": error.max().item(),
                "bitwise_equal": torch.equal(output, reference),
            })
            del output, error

    print(json.dumps({
        "scope": "synthetic single-module benchmark; no checkpoint or end-to-end evidence",
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "dtype": args.dtype,
        "shape": {"segments": args.segments, "length": args.sequence_length,
                  "hidden_size": args.hidden_size, "heads": args.heads},
        "warmup_per_batch_size": args.warmup,
        "memory_scope": "total CUDA allocated peak including model, inputs and retained baseline output",
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
