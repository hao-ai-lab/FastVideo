# SPDX-License-Identifier: Apache-2.0
"""Apple-Silicon accelerator probe for the MLX denoise path.

Answers one question before we trust any M5 speedup number: *is the Metal 4
Neural-Accelerator path actually engaged, or did MLX silently fall back?* It
prints the platform, then microbenchmarks the two shapes that dominate our Wan
DiT step — a large linear GEMM and the long-sequence fused attention — in fp16
and in each supported quantized backend, and reports the quantized-vs-fp16 speed
ratio. On an M5 the int8/mxfp8 GEMM should run materially faster than fp16
(dedicated matrix units); on pre-M5 silicon the ratio sits near ~1.0.

Run identically on M4 and M5 and diff the JSON:

    python -m fastvideo.mlx_runtime.accel_probe --json bench/accel/m4.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx

from fastvideo.mlx_runtime.quant_backends import (
    BACKENDS,
    is_supported,
    quantize_weight,
    quantized_matmul,
)

# Defaults mirror a real 1.3B Wan step at 480x832x81f: ~32,760-token sequence,
# 12 heads x 128 head-dim, and a representative FFN-scale square GEMM.
_DEFAULT_GEMM: tuple[int, int, int] = (4096, 4096, 4096)
_DEFAULT_ATTN: tuple[int, int, int, int] = (1, 12, 32760, 128)


@dataclass(frozen=True)
class PlatformInfo:
    """Static description of the machine and MLX build under test."""

    chip: str
    macos: str
    mlx_version: str
    device: str
    neural_accel_expected: bool


@dataclass(frozen=True)
class BenchResult:
    """One microbenchmark: a shape run under one backend."""

    kind: str            # "gemm" | "attention"
    backend: str         # "fp16" or a QuantBackend value
    shape: tuple[int, ...]
    iters: int
    seconds_per_iter: float
    tflops: float
    speedup_vs_fp16: float | None


def _sysctl(name: str) -> str:
    try:
        out = subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, timeout=5, check=True
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def detect_platform() -> PlatformInfo:
    """Read chip / macOS / MLX identity and guess whether M5-class accel exists."""
    chip = _sysctl("machdep.cpu.brand_string")
    macos = platform.mac_ver()[0] or "unknown"
    mlx_version = str(getattr(mx, "__version__", "unknown"))
    device = str(mx.default_device())
    # Neural Accelerators ship on M5 / A19 GPUs; earlier chips lack them.
    accel = any(tag in chip for tag in ("M5", "A19"))
    return PlatformInfo(
        chip=chip,
        macos=macos,
        mlx_version=mlx_version,
        device=device,
        neural_accel_expected=accel,
    )


def _time_call(fn, warmup: int, iters: int) -> float:
    """Return median-free mean seconds per call, forcing evaluation each time."""
    for _ in range(warmup):
        mx.eval(fn())
    start = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    return (time.perf_counter() - start) / iters


def bench_gemm(
    backend: str, m: int, k: int, n: int, *, warmup: int, iters: int
) -> BenchResult:
    """Benchmark ``(m,k) @ (k,n)`` in fp16 or a quantized backend.

    FLOPs for a dense matmul are ``2*m*k*n``; we report that against wall time
    regardless of backend so fp16 and quantized paths are directly comparable.
    """
    x = mx.random.normal((m, k)).astype(mx.float16)
    flops = 2.0 * m * k * n
    if backend == "fp16":
        w = mx.random.normal((n, k)).astype(mx.float16)

        def run() -> mx.array:
            return x @ w.T
    else:
        w = mx.random.normal((n, k)).astype(mx.float16)
        qw = quantize_weight(w, backend)

        def run() -> mx.array:
            return quantized_matmul(x, qw)

    spi = _time_call(run, warmup, iters)
    return BenchResult(
        kind="gemm",
        backend=backend,
        shape=(m, k, n),
        iters=iters,
        seconds_per_iter=spi,
        tflops=flops / spi / 1e12,
        speedup_vs_fp16=None,
    )


def bench_attention(
    b: int, h: int, s: int, d: int, *, warmup: int, iters: int
) -> BenchResult:
    """Benchmark the fused ``mx.fast.scaled_dot_product_attention`` at our shape."""
    scale = 1.0 / (d ** 0.5)
    q = mx.random.normal((b, h, s, d)).astype(mx.float16)
    k = mx.random.normal((b, h, s, d)).astype(mx.float16)
    v = mx.random.normal((b, h, s, d)).astype(mx.float16)

    def run() -> mx.array:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)

    # QK^T and AV are each ~2*b*h*s*s*d FLOPs.
    flops = 4.0 * b * h * s * s * d
    spi = _time_call(run, warmup, iters)
    return BenchResult(
        kind="attention",
        backend="fp16",
        shape=(b, h, s, d),
        iters=iters,
        seconds_per_iter=spi,
        tflops=flops / spi / 1e12,
        speedup_vs_fp16=None,
    )


def _with_speedups(results: list[BenchResult]) -> list[BenchResult]:
    """Fill ``speedup_vs_fp16`` for GEMM rows relative to the fp16 GEMM baseline."""
    fp16_gemm = next(
        (r for r in results if r.kind == "gemm" and r.backend == "fp16"), None
    )
    if fp16_gemm is None:
        return results
    base = fp16_gemm.seconds_per_iter
    out: list[BenchResult] = []
    for r in results:
        if r.kind == "gemm" and r.backend != "fp16":
            ratio = base / r.seconds_per_iter if r.seconds_per_iter > 0 else None
            out.append(BenchResult(**{**asdict(r), "speedup_vs_fp16": ratio}))
        else:
            out.append(r)
    return out


def run_probe(
    *, gemm: tuple[int, int, int], attn: tuple[int, int, int, int], warmup: int, iters: int
) -> dict:
    """Run all microbenchmarks and return a JSON-serializable report."""
    info = detect_platform()
    results: list[BenchResult] = [bench_gemm("fp16", *gemm, warmup=warmup, iters=iters)]
    for backend in BACKENDS:
        if is_supported(backend):
            results.append(bench_gemm(backend, *gemm, warmup=warmup, iters=iters))
    results.append(bench_attention(*attn, warmup=warmup, iters=iters))
    results = _with_speedups(results)

    best_quant = max(
        (r.speedup_vs_fp16 for r in results if r.speedup_vs_fp16 is not None),
        default=None,
    )
    # Weight-only quantization already speeds a GEMM ~1.5-1.8x on pre-M5 silicon
    # purely from reading fewer weight bytes (measured on M4 Max). So a quant
    # speedup alone does NOT prove the Neural Accelerator is engaged. We only
    # flag it when the chip is M5-class AND the quant GEMM clears that bandwidth
    # ceiling by a wide margin (dedicated matrix units should push it well past
    # ~2.5x). This is a coarse gate; the authoritative signal is diffing this
    # JSON between an M4 and an M5 run.
    _M4_BANDWIDTH_CEILING = 2.5
    engaged = bool(
        info.neural_accel_expected
        and best_quant is not None
        and best_quant > _M4_BANDWIDTH_CEILING
    )
    return {
        "platform": asdict(info),
        "results": [asdict(r) for r in results],
        "best_quant_gemm_speedup_vs_fp16": best_quant,
        "accelerator_likely_engaged": engaged,
    }


def _print_report(report: dict) -> None:
    p = report["platform"]
    print(f"chip={p['chip']}  macOS={p['macos']}  mlx={p['mlx_version']}  device={p['device']}")
    print(f"neural_accel_expected={p['neural_accel_expected']}")
    print(f"{'kind':<10} {'backend':<16} {'shape':<22} {'s/iter':>10} {'TFLOP/s':>10} {'x fp16':>8}")
    for r in report["results"]:
        sp = "" if r["speedup_vs_fp16"] is None else f"{r['speedup_vs_fp16']:.2f}"
        shape = "x".join(str(v) for v in r["shape"])
        print(
            f"{r['kind']:<10} {r['backend']:<16} {shape:<22} "
            f"{r['seconds_per_iter']*1e3:>9.2f}m {r['tflops']:>10.2f} {sp:>8}"
        )
    best = report["best_quant_gemm_speedup_vs_fp16"]
    best_s = "n/a" if best is None else f"{best:.2f}x"
    print(f"\nbest quantized GEMM speedup vs fp16: {best_s}")
    print(f"accelerator_likely_engaged: {report['accelerator_likely_engaged']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX Apple-Silicon accelerator probe.")
    parser.add_argument("--gemm", default=",".join(str(v) for v in _DEFAULT_GEMM),
                        help="M,K,N for the linear GEMM benchmark.")
    parser.add_argument("--attn", default=",".join(str(v) for v in _DEFAULT_ATTN),
                        help="B,H,S,D for the attention benchmark.")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--json", type=Path, default=None, help="Write the report here.")
    args = parser.parse_args()

    gemm = tuple(int(v) for v in args.gemm.split(","))
    attn = tuple(int(v) for v in args.attn.split(","))
    if len(gemm) != 3 or len(attn) != 4:
        raise SystemExit("--gemm needs 3 values (M,K,N); --attn needs 4 (B,H,S,D)")

    report = run_probe(gemm=gemm, attn=attn, warmup=args.warmup, iters=args.iters)  # type: ignore[arg-type]
    _print_report(report)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
