# SPDX-License-Identifier: Apache-2.0
"""Full vs windowed attention scaling probe for the MLX Wan denoise shape.

Measures wall-clock and approximation quality of chunked symmetric sliding-window
attention against dense ``mx.fast.scaled_dot_product_attention`` at the shapes
that dominate a Wan DiT step (B=1, H=12, D=128, long S).

Example::

    python -m fastvideo.mlx_runtime.attn_scaling_probe \\
        --json bench/accel/attn_m4.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx

from fastvideo.mlx_runtime.windowed_attention import full_attention, windowed_attention

# Defaults mirror a real 1.3B Wan step at ~480x832x81f: 12 heads x 128 dim.
_DEFAULT_B = 1
_DEFAULT_H = 12
_DEFAULT_D = 128
_DEFAULT_SEQS = (8192, 16384, 32760)
_DEFAULT_WINDOWS = (1024, 2048, 4096)
_SEED = 20260710


@dataclass(frozen=True)
class Row:
    """One (S, window) measurement row."""

    seq_len: int
    window: int
    sink: int
    full_seconds: float
    windowed_seconds: float
    speedup: float
    mean_cosine: float
    shape: tuple[int, int, int, int]


def _sysctl(name: str) -> str:
    try:
        out = subprocess.run(
            ["sysctl", "-n", name],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        return f"unknown ({type(exc).__name__})"


def _time_call(fn, warmup: int, iters: int) -> float:
    """Mean seconds per call; force Metal completion with ``mx.eval`` each iter."""
    if warmup < 0 or iters < 1:
        raise ValueError(f"warmup >= 0 and iters >= 1 required; got {warmup=}, {iters=}")
    for _ in range(warmup):
        mx.eval(fn())
    start = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    return (time.perf_counter() - start) / iters


def mean_cosine_similarity(a: mx.array, b: mx.array) -> float:
    """Mean cosine similarity over all ``(B, H, S)`` head vectors (fp32)."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch for cosine: {a.shape} vs {b.shape}")
    a32 = a.astype(mx.float32).reshape(-1, a.shape[-1])
    b32 = b.astype(mx.float32).reshape(-1, b.shape[-1])
    num = mx.sum(a32 * b32, axis=-1)
    den = mx.sqrt(mx.sum(a32 * a32, axis=-1) * mx.sum(b32 * b32, axis=-1))
    # Guard zero-norm rows (should not appear for random Gaussian q/k/v outputs).
    cos = num / mx.maximum(den, 1e-12)
    mx.eval(cos)
    return float(mx.mean(cos).item())


def _make_qkv(
    b: int, h: int, s: int, d: int, *, seed: int
) -> tuple[mx.array, mx.array, mx.array]:
    mx.random.seed(seed)
    q = mx.random.normal((b, h, s, d)).astype(mx.float16)
    k = mx.random.normal((b, h, s, d)).astype(mx.float16)
    v = mx.random.normal((b, h, s, d)).astype(mx.float16)
    mx.eval(q, k, v)
    return q, k, v


def bench_pair(
    *,
    b: int,
    h: int,
    s: int,
    d: int,
    window: int,
    sink: int,
    warmup: int,
    iters: int,
    seed: int,
) -> Row:
    """Time full vs windowed attention and measure mean cosine on fixed q/k/v."""
    q, k, v = _make_qkv(b, h, s, d, seed=seed)
    scale = d**-0.5

    def run_full() -> mx.array:
        return full_attention(q, k, v, scale=scale)

    def run_win() -> mx.array:
        return windowed_attention(q, k, v, window=window, sink=sink, scale=scale)

    full_s = _time_call(run_full, warmup, iters)
    win_s = _time_call(run_win, warmup, iters)

    # Approximation error on the same fixed tensors (not re-sampled).
    out_full = full_attention(q, k, v, scale=scale)
    out_win = windowed_attention(q, k, v, window=window, sink=sink, scale=scale)
    mx.eval(out_full, out_win)
    cos = mean_cosine_similarity(out_full, out_win)

    speedup = full_s / win_s if win_s > 0 else float("inf")
    return Row(
        seq_len=s,
        window=window,
        sink=sink,
        full_seconds=full_s,
        windowed_seconds=win_s,
        speedup=speedup,
        mean_cosine=cos,
        shape=(b, h, s, d),
    )


def format_table(rows: list[Row]) -> str:
    """Pretty fixed-width table for stdout."""
    headers = (
        "S",
        "window",
        "sink",
        "full_s",
        "win_s",
        "speedup",
        "mean_cos",
    )
    body: list[tuple[str, ...]] = []
    for r in rows:
        body.append(
            (
                str(r.seq_len),
                str(r.window),
                str(r.sink),
                f"{r.full_seconds:.4f}",
                f"{r.windowed_seconds:.4f}",
                f"{r.speedup:.2f}x",
                f"{r.mean_cosine:.6f}",
            )
        )
    cols = list(zip(*([headers] + body), strict=True))
    widths = [max(len(cell) for cell in col) for col in cols]

    def fmt_row(cells: tuple[str, ...]) -> str:
        return "  ".join(c.rjust(w) for c, w in zip(cells, widths, strict=True))

    lines = [fmt_row(headers), "  ".join("-" * w for w in widths)]
    lines.extend(fmt_row(row) for row in body)
    return "\n".join(lines)


def run_probe(
    *,
    seqs: tuple[int, ...],
    windows: tuple[int, ...],
    b: int = _DEFAULT_B,
    h: int = _DEFAULT_H,
    d: int = _DEFAULT_D,
    sink: int = 0,
    warmup: int = 2,
    iters: int = 5,
    seed: int = _SEED,
) -> dict:
    """Run the full grid and return a JSON-serializable report."""
    if not seqs or not windows:
        raise ValueError("seqs and windows must be non-empty")

    rows: list[Row] = []
    for s in seqs:
        for w in windows:
            print(f"[probe] S={s} window={w} sink={sink} ...", flush=True)
            row = bench_pair(
                b=b,
                h=h,
                s=s,
                d=d,
                window=w,
                sink=sink,
                warmup=warmup,
                iters=iters,
                seed=seed + s + w,
            )
            rows.append(row)
            print(
                f"         full={row.full_seconds:.4f}s  "
                f"win={row.windowed_seconds:.4f}s  "
                f"speedup={row.speedup:.2f}x  "
                f"cos={row.mean_cosine:.6f}",
                flush=True,
            )

    report = {
        "platform": {
            "chip": _sysctl("machdep.cpu.brand_string"),
            "macos": platform.mac_ver()[0] or "unknown",
            "mlx_version": str(getattr(mx, "__version__", "unknown")),
            "device": str(mx.default_device()),
        },
        "config": {
            "B": b,
            "H": h,
            "D": d,
            "dtype": "float16",
            "seqs": list(seqs),
            "windows": list(windows),
            "sink": sink,
            "warmup": warmup,
            "iters": iters,
            "seed_base": seed,
            "window_policy": "symmetric (half = window // 2) + optional global sinks",
            "implementation": "chunked SDPA over local key slices (not full additive mask)",
        },
        "rows": [asdict(r) for r in rows],
    }
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark full vs chunked windowed attention (MLX Wan shapes)."
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write the full JSON report to this path.",
    )
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=5, help="Timed iterations.")
    parser.add_argument(
        "--sink",
        type=int,
        default=0,
        help="Global sink token count for windowed attention (default 0).",
    )
    parser.add_argument(
        "--seqs",
        type=int,
        nargs="+",
        default=list(_DEFAULT_SEQS),
        help="Sequence lengths to sweep.",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=list(_DEFAULT_WINDOWS),
        help="Window sizes to sweep.",
    )
    args = parser.parse_args(argv)

    if args.warmup < 0:
        raise SystemExit(f"--warmup must be >= 0, got {args.warmup}")
    if args.iters < 1:
        raise SystemExit(f"--iters must be >= 1, got {args.iters}")
    if args.sink < 0:
        raise SystemExit(f"--sink must be >= 0, got {args.sink}")

    report = run_probe(
        seqs=tuple(args.seqs),
        windows=tuple(args.windows),
        sink=args.sink,
        warmup=args.warmup,
        iters=args.iters,
    )
    rows = [Row(**{k: (tuple(v) if k == "shape" else v) for k, v in r.items()}) for r in report["rows"]]

    print()
    print("=== Full vs windowed attention (B=1, H=12, D=128, fp16) ===")
    print(f"chip={report['platform']['chip']}  mlx={report['platform']['mlx_version']}")
    print(f"policy={report['config']['window_policy']}")
    print()
    print(format_table(rows))
    print()

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote JSON report to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
