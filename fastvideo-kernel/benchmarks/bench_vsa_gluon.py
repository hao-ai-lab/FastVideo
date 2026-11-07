# SPDX-License-Identifier: Apache-2.0
"""Compare native Triton and Gluon sparse forwards on identical compact indices."""

import argparse
import json
import statistics
import traceback
from datetime import datetime
from pathlib import Path

import torch
import triton

from fastvideo_kernel.block_sparse_attn import block_sparse_attn_gluon_fwd, block_sparse_attn_triton
from fastvideo_kernel.triton_kernels.index import map_to_index


def make_inputs(args):
    if args.capture:
        capture = torch.load(args.capture, map_location="cpu", weights_only=True)
        q, k, v = (capture[name].to("cuda").contiguous() for name in ("q", "k", "v"))
        mask = capture["mask"].to(device="cuda", dtype=torch.bool).contiguous()
        sizes = capture["vbs"].to(device="cuda", dtype=torch.int32).contiguous()
        if mask.shape != (q.shape[0], q.shape[1], q.shape[2] // 64, k.shape[2] // 64):
            raise ValueError("Capture mask does not match BHSD Q/K shapes")
        indices, counts = map_to_index(mask)
        return q, k, v, indices, counts, sizes
    torch.manual_seed(args.seed)
    shape = (1, args.heads, args.tiles * 64, 128)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    sizes = torch.full((args.tiles,), 64, device="cuda", dtype=torch.int32)
    video_tiles = args.tiles - args.prefix_tiles
    topk = max(1, int(video_tiles * (1 - args.sparsity)))
    scores = torch.rand((1, args.heads, video_tiles, video_tiles), device="cuda")
    selected = scores.topk(topk, -1).indices.sort(-1).values.int() + args.prefix_tiles
    del scores
    indices = torch.zeros((1, args.heads, args.tiles, args.tiles), device="cuda", dtype=torch.int32)
    prefix = torch.arange(args.prefix_tiles, device="cuda", dtype=torch.int32)
    indices[:, :, args.prefix_tiles:, :args.prefix_tiles] = prefix
    indices[:, :, args.prefix_tiles:, args.prefix_tiles:args.prefix_tiles + topk] = selected
    indices[:, :, :args.prefix_tiles] = torch.arange(args.tiles, device="cuda", dtype=torch.int32)
    counts = torch.full(shape[:2] + (args.tiles,), args.prefix_tiles + topk, device="cuda", dtype=torch.int32)
    counts[:, :, :args.prefix_tiles] = args.tiles
    return q, k, v, indices, counts, sizes


def validate_inputs(inputs):
    q, k, _, indices, counts, sizes = inputs
    if (counts < 0).any() or (counts > indices.shape[-1]).any():
        raise ValueError("Selected counts exceed index capacity")
    selected = torch.arange(indices.shape[-1], device=q.device) < counts[..., None]
    if ((indices < 0) | (indices >= k.shape[2] // 64))[selected].any():
        raise ValueError("Selected KV index is out of bounds")
    if (sizes < 1).any() or (sizes > 64).any():
        raise ValueError("The native comparison requires valid KV sizes in [1, 64]")
    valid = torch.arange(k.shape[2], device=q.device) % 64 < sizes.repeat_interleave(64)
    if not torch.isfinite(inputs[2][:, :, ~valid]).all():
        raise ValueError("The native comparison requires finite value padding")


def compare(actual, expected, atol, rtol):
    largest, squared, count = 0.0, 0.0, 0
    a, b = actual.flatten(), expected.flatten()
    for start in range(0, a.numel(), 1 << 20):
        x, y = a[start:start + (1 << 20)].float(), b[start:start + (1 << 20)].float()
        torch.testing.assert_close(x, y, atol=atol, rtol=rtol)
        finite = torch.isfinite(x) & torch.isfinite(y)
        error = (x[finite] - y[finite]).double()
        if error.numel():
            largest = max(largest, error.abs().max().item())
            squared += error.square().sum().item()
            count += error.numel()
    return {"max_absolute_error": largest, "rmse": (squared / max(1, count))**0.5, "atol": atol, "rtol": rtol}


def measure(operation, inputs):
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    output = operation(*inputs)
    end.record()
    end.synchronize()
    elapsed = start.elapsed_time(end)
    extra = torch.cuda.max_memory_allocated() - allocated
    del output
    return elapsed, extra


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--tiles", type=int, default=1802)
    parser.add_argument("--prefix-tiles", type=int, default=21)
    parser.add_argument("--sparsity", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--capture", type=Path, help="Trusted .pt containing BHSD q/k/v, mask and vbs")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[3] / "outputs/profiles" /
                        datetime.now().strftime("%Y%m%d_%H%M%S") / "result.json")
    args = parser.parse_args()
    if (args.heads < 1 or not 0 <= args.prefix_tiles < args.tiles or not 0 <= args.sparsity < 1
            or args.repeats < 1 or args.warmup < 1):
        parser.error("Require positive heads/repeats/warmup, prefix_tiles < tiles and sparsity in [0, 1)")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This benchmark requires an SM120 CUDA GPU")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".log").open("x", encoding="utf-8") as log:
        def record(message):
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()

        try:
            record("Preparing identical inputs for native Triton and Gluon async_overlap2 on SM120.")
            inputs = make_inputs(args)
            validate_inputs(inputs)
            operations = {"native_triton": block_sparse_attn_triton, "gluon": block_sparse_attn_gluon_fwd}
            record("Compiling and checking numerical output and log2-LSE; excluded from timing.")
            reference = operations["native_triton"](*inputs)
            actual = operations["gluon"](*inputs)
            numerical = {"output": compare(actual[0], reference[0], 0.02, 0.02),
                         "log2_lse": compare(actual[1], reference[1], 0.005, 0.001)}
            del reference, actual
            for operation in operations.values():
                for _ in range(args.warmup):
                    operation(*inputs)
            torch.cuda.synchronize()
            timings = {name: [] for name in operations}
            peaks = {name: 0 for name in operations}
            names = list(operations)
            for iteration in range(args.repeats):
                for name in names[::1 if iteration % 2 == 0 else -1]:
                    elapsed, peak = measure(operations[name], inputs)
                    timings[name].append(elapsed)
                    peaks[name] = max(peaks[name], peak)
                record(f"Round {iteration + 1}/{args.repeats}: " + ", ".join(
                    f"{name}={timings[name][-1]:.3f} ms" for name in names))
            medians = {name: statistics.median(values) for name, values in timings.items()}
            report = {
                "scope": "Sparse forward only; excludes mask/index construction, model loading and layout conversion.",
                "input_source": (str(args.capture.resolve()) if args.capture
                                 else "synthetic (H3 prefix/dense-query policy)"),
                "gpu": torch.cuda.get_device_name(),
                "torch_version": torch.__version__, "triton_version": triton.__version__,
                "q_shape": list(inputs[0].shape), "k_shape": list(inputs[1].shape), "seed": args.seed,
                "synthetic_sparsity": args.sparsity if not args.capture else None,
                "warmup": args.warmup, "repeats": args.repeats, "numerical": numerical,
                "cuda_median_ms": medians, "cuda_samples_ms": timings, "peak_extra_allocated_bytes": peaks,
                "attention_speedup": medians["native_triton"] / medians["gluon"],
                "attention_time_reduction_pct": 100 * (1 - medians["gluon"] / medians["native_triton"]),
            }
            with args.output.open("x", encoding="utf-8") as output:
                json.dump(report, output, indent=2, allow_nan=False)
                output.write("\n")
            record(json.dumps(report, indent=2, allow_nan=False))
            record(f"Saved: {args.output.resolve()}")
        except BaseException:
            traceback.print_exc(file=log)
            log.flush()
            raise


if __name__ == "__main__":
    main()
