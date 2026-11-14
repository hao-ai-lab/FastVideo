# SPDX-License-Identifier: Apache-2.0
"""Compare original per-frame SDPA and packed PyTorch varlen attention.

Run from the repository root with PYTHONPATH=. in a prepared FastVideo environment::

    python tests/local_tests/encoders/benchmark_minimax_h3_vision_attention.py \
        --device cuda --dtype bfloat16 --segments 16 --sequence-length 1024

Use --sequence-lengths 512 1024 768 to test unequal independent frames.
This measures one native vision-attention module, including QKV, RoPE and output
projection. It does not measure checkpoint quality or end-to-end H3 latency.
Packed metadata is built outside timing, matching its reuse across vision blocks.
Each mode uses identical weights and inputs; timings exclude setup and warmup.
An untimed call records attention calls without retaining input tensors.
Native varlen requires a compatible CUDA FP16/BF16 runtime. CPU/FP32 smoke checks
run only the original baseline. If the candidate selects its compatibility
fallback, this benchmark reports a skipped varlen comparison.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import statistics
import time
from unittest.mock import patch

import torch

import fastvideo.models.encoders.minimax_h3_qwen3_vl as vision
from fastvideo.configs.models.encoders.minimax_h3_qwen3_vl import (
    MiniMaxH3Qwen3VLArchConfig,
    MiniMaxH3Qwen3VLConfig,
)
from fastvideo.models.encoders.minimax_h3_qwen3_vl import (
    MiniMaxH3Qwen3VLVisionAttention,
    _vision_attention_metadata,
)
from torch.nn.attention import SDPBackend, sdpa_kernel


def _looped_attention(model, hidden, lengths, positions):
    """Original attention computation, kept here only as the benchmark baseline."""
    rows = hidden.shape[0]
    query, key, value = model.qkv(hidden).view(rows, 3, model.num_heads, model.head_dim).unbind(1)
    cos, sin = (part.unsqueeze(-2).float() for part in positions)

    def rotate(tensor):
        left, right = tensor.float().chunk(2, dim=-1)
        return (tensor.float() * cos + torch.cat((-right, left), dim=-1) * sin).to(tensor.dtype)

    query, key = rotate(query), rotate(key)
    outputs = []
    offset = 0
    for length in lengths:
        end = offset + length
        output = vision.F.scaled_dot_product_attention(
            query[offset:end].transpose(0, 1).unsqueeze(0),
            key[offset:end].transpose(0, 1).unsqueeze(0),
            value[offset:end].transpose(0, 1).unsqueeze(0),
            dropout_p=0.0, is_causal=False, scale=model.scaling,
        )
        outputs.append(output.transpose(1, 2))
        offset = end
    return model.proj(torch.cat(outputs, dim=1).reshape(rows, -1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--segments", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--sequence-lengths", type=int, nargs="+", help="Overrides --segments and --sequence-length")
    parser.add_argument("--hidden-size", type=int, default=1152)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--modes", choices=("loop", "varlen"), nargs="+", default=["loop", "varlen"])
    parser.add_argument("--sdpa-backend", choices=("auto", "math", "flash", "efficient", "cudnn"),
                        default="auto", help="Controls only the original SDPA baseline")
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
    template = MiniMaxH3Qwen3VLVisionAttention(config)
    weights = {name: tensor.detach().clone() for name, tensor in template.state_dict().items()}
    del template
    rows = sum(lengths)
    hidden = torch.randn(rows, args.hidden_size, device=device, dtype=dtype)
    angles = torch.randn(rows, args.hidden_size // args.heads // 2, device=device)
    angles = torch.cat((angles, angles), dim=-1)
    positions = (angles.cos(), angles.sin())
    modes = list(dict.fromkeys(["loop", *args.modes]))
    sdpa_backends = {"math": SDPBackend.MATH, "flash": SDPBackend.FLASH_ATTENTION,
                     "efficient": SDPBackend.EFFICIENT_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION}

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    results = []
    reference = None
    with torch.inference_mode():
        for mode in modes:
            if mode == "varlen" and (device.type != "cuda" or dtype not in (torch.float16, torch.bfloat16)):
                results.append({"mode": mode, "status": "skipped",
                                "reason": "Native varlen comparison requires compatible CUDA FP16/BF16"})
                continue
            model = MiniMaxH3Qwen3VLVisionAttention(config)
            model.load_state_dict(weights)
            model = model.to(device=device, dtype=dtype).eval()
            metadata = _vision_attention_metadata(lengths, rows, device) if mode == "varlen" else None

            def run() -> torch.Tensor:
                if mode == "varlen":
                    return model(hidden, lengths, positions, metadata)
                context = (sdpa_kernel(sdpa_backends[args.sdpa_backend])
                           if args.sdpa_backend != "auto" else nullcontext())
                with context:
                    return _looped_attention(model, hidden, lengths, positions)

            # Record only scalar metadata. Mock argument records would retain
            # setup Q/K/V and inflate the later CUDA memory peak.
            sdpa_batch_sizes = []
            selected_backends = []
            original_sdpa = vision.F.scaled_dot_product_attention

            def recorded_sdpa(query, key, value, **kwargs):
                sdpa_batch_sizes.append(query.shape[0])
                selector = getattr(torch, "_fused_sdp_choice", None)
                if callable(selector):
                    try:
                        selected = SDPBackend(selector(query, key, value, **kwargs)).name
                    except torch.OutOfMemoryError:
                        raise
                    except (RuntimeError, TypeError, ValueError):
                        selected = "unknown"
                else:
                    selected = "unknown"
                selected_backends.append(selected)
                return original_sdpa(query, key, value, **kwargs)

            varlen_calls = 0
            original_varlen = vision.varlen_attn

            def recorded_varlen(*varlen_args, **varlen_kwargs):
                nonlocal varlen_calls
                varlen_calls += 1
                return original_varlen(*varlen_args, **varlen_kwargs)

            with patch.object(vision.F, "scaled_dot_product_attention", new=recorded_sdpa):
                with patch.object(vision, "varlen_attn", new=recorded_varlen):
                    setup_output = run()
            del recorded_sdpa, recorded_varlen
            if mode == "varlen" and varlen_calls == 0 and sdpa_batch_sizes:
                results.append({"mode": mode, "status": "skipped",
                                "reason": "PyTorch Flash eligibility selected the per-frame SDPA fallback"})
                del setup_output, model, metadata
                continue
            if mode == "varlen" and (varlen_calls != 1 or sdpa_batch_sizes):
                raise RuntimeError("Varlen comparison must execute exactly one native varlen call and no SDPA loop")
            if reference is None:
                reference = setup_output
            del setup_output
            for _ in range(args.warmup):
                warm_output = run()
                del warm_output
            synchronize()
            start_allocated = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
            start_reserved = torch.cuda.memory_reserved(device) if device.type == "cuda" else None
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
            peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
            output = run()
            error = (output.float() - reference.float()).abs()
            tolerance = {torch.bfloat16: (1e-2, 3e-3), torch.float16: (2e-3, 3e-4),
                         torch.float32: (1e-4, 1e-5)}[dtype]
            torch.testing.assert_close(output, reference, rtol=tolerance[0], atol=tolerance[1])
            results.append({
                "mode": mode,
                "status": "measured",
                "execution": "packed_varlen" if mode == "varlen" else "per_segment",
                "attention_calls_per_forward": varlen_calls + len(sdpa_batch_sizes),
                "observed_varlen_calls": varlen_calls,
                "observed_sdpa_batch_sizes": sdpa_batch_sizes,
                "setup_predicted_backends": selected_backends,
                "sdpa_backend_setting": args.sdpa_backend if mode == "loop" else None,
                "median_wall_ms": statistics.median(timings),
                "wall_ms": timings,
                "starting_allocated_bytes": start_allocated,
                "peak_allocated_bytes": peak,
                "extra_peak_allocated_bytes": peak - start_allocated if peak is not None else None,
                "starting_reserved_bytes": start_reserved,
                "peak_reserved_bytes": peak_reserved,
                "max_abs_error": error.max().item(),
                "bitwise_equal": torch.equal(output, reference),
            })
            del output, error, model, metadata

    print(json.dumps({
        "scope": "synthetic single-module benchmark; no checkpoint or end-to-end evidence",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "dtype": args.dtype,
        "shape": {"sequence_lengths": lengths,
                  "hidden_size": args.hidden_size, "heads": args.heads},
        "warmup_per_mode": args.warmup,
        "untimed_setup_calls_per_mode": 1,
        "memory_scope": "allocated peaks include one model, inputs and retained baseline output; "
                        "reserved peaks also include allocator cache from earlier modes",
        "receipt_scope": "observed varlen/SDPA calls; backend predictions apply to SDPA baseline only",
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
