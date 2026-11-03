# SPDX-License-Identifier: Apache-2.0
"""Compare H3 vision SDPA and packed FA2 using synthetic inputs and random weights.

Run from the repository root with PYTHONPATH=. in a prepared FastVideo environment::

    python tests/local_tests/encoders/benchmark_minimax_h3_vision_attention.py \
        --device cuda --dtype bfloat16 --segments 16 --sequence-length 1024

Use --sequence-lengths 512 1024 768 to test unequal independent frames.
This measures one native vision-attention module, including QKV, RoPE and output
projection. It does not measure checkpoint quality or end-to-end H3 latency.
Metadata is built outside timing, matching its reuse across the vision blocks.
Each backend uses identical weights and inputs; timings exclude warmup.
An untimed setup call on each backend also excludes lazy import/kernel setup.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.metadata
import json
import statistics
import time

import torch

from fastvideo.configs.models.encoders.minimax_h3_qwen3_vl import (
    MiniMaxH3Qwen3VLArchConfig,
    MiniMaxH3Qwen3VLConfig,
)
from fastvideo.models.encoders.minimax_h3_qwen3_vl import (
    MiniMaxH3Qwen3VLVisionAttention,
    _can_use_vision_flash_attn_2,
    _vision_attention_metadata,
)
from torch.nn.attention import SDPBackend, sdpa_kernel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--segments", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--sequence-lengths", type=int, nargs="+", help="Overrides --segments and --sequence-length")
    parser.add_argument("--hidden-size", type=int, default=1152)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--backends", choices=("sdpa", "flash_attn_2"), nargs="+", default=["sdpa", "flash_attn_2"])
    parser.add_argument("--sdpa-backend", choices=("auto", "math", "flash", "efficient"), default="auto")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    lengths = args.sequence_lengths if args.sequence_lengths is not None else [args.sequence_length] * args.segments
    positive = [args.segments, args.sequence_length, args.hidden_size, args.heads, args.runs, *lengths]
    if any(value < 1 for value in positive) or args.warmup < 0:
        parser.error("dimensions and runs must be positive; warmup must be non-negative")
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
    template = MiniMaxH3Qwen3VLVisionAttention(config, attn_implementation="sdpa")
    weights = {name: tensor.detach().clone() for name, tensor in template.state_dict().items()}
    del template
    rows = sum(lengths)
    hidden = torch.randn(rows, args.hidden_size, device=device, dtype=dtype)
    angles = torch.randn(rows, args.hidden_size // args.heads // 2, device=device)
    angles = torch.cat((angles, angles), dim=-1)
    positions = (angles.cos(), angles.sin())
    backends = list(dict.fromkeys(["sdpa", *args.backends]))
    sdpa_backends = {"math": SDPBackend.MATH, "flash": SDPBackend.FLASH_ATTENTION,
                     "efficient": SDPBackend.EFFICIENT_ATTENTION}

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    results = []
    reference = None
    with torch.inference_mode():
        for backend in backends:
            if backend == "flash_attn_2" and not _can_use_vision_flash_attn_2(hidden, args.hidden_size // args.heads):
                results.append({"backend": backend, "status": "skipped",
                                "reason": "Inputs are ineligible for FA2; refusing to label an SDPA fallback as FA2"})
                continue
            model = MiniMaxH3Qwen3VLVisionAttention(config, attn_implementation=backend)
            model.load_state_dict(weights)
            model = model.to(device=device, dtype=dtype).eval()
            metadata = _vision_attention_metadata(lengths, device) if backend == "flash_attn_2" else None

            def run() -> torch.Tensor:
                context = (sdpa_kernel(sdpa_backends[args.sdpa_backend])
                           if backend == "sdpa" and args.sdpa_backend != "auto" else nullcontext())
                with context:
                    return model(hidden, lengths, positions, metadata)

            setup_output = run()
            if reference is None:
                reference = setup_output
            del setup_output
            for _ in range(args.warmup):
                warm_output = run()
                del warm_output
            synchronize()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            timings = []
            for _ in range(args.runs):
                synchronize()
                start = time.perf_counter()
                output = run()
                synchronize()
                timings.append((time.perf_counter() - start) * 1000)
                del output
            peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            # Compare outside the timing and memory-measurement interval.
            output = run()
            error = (output.float() - reference.float()).abs()
            tolerance = {torch.bfloat16: (1e-2, 3e-3), torch.float16: (2e-3, 3e-4),
                         torch.float32: (1e-4, 1e-5)}[dtype]
            torch.testing.assert_close(output, reference, rtol=tolerance[0], atol=tolerance[1])
            results.append({
                "backend": backend,
                "status": "measured",
                "attention_calls_per_forward": len(lengths) if backend == "sdpa" else 1,
                "sdpa_backend": args.sdpa_backend if backend == "sdpa" else None,
                "median_wall_ms": statistics.median(timings),
                "wall_ms": timings,
                "peak_allocated_bytes": peak,
                "max_abs_error": error.max().item(),
                "bitwise_equal": torch.equal(output, reference),
            })
            del output, error, model, metadata

    try:
        flash_attn_version = importlib.metadata.version("flash-attn")
    except importlib.metadata.PackageNotFoundError:
        flash_attn_version = None

    print(json.dumps({
        "scope": "synthetic single-module benchmark; no checkpoint or end-to-end evidence",
        "torch_version": torch.__version__,
        "flash_attn_version": flash_attn_version,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "dtype": args.dtype,
        "shape": {"sequence_lengths": lengths,
                  "hidden_size": args.hidden_size, "heads": args.heads},
        "warmup_per_backend": args.warmup,
        "untimed_setup_calls_per_backend": 1,
        "memory_scope": "total CUDA allocated peak including model, inputs and retained baseline output",
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
