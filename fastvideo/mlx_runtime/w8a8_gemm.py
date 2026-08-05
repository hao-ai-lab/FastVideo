# SPDX-License-Identifier: Apache-2.0
"""W8A8 fused INT8 GEMM prototype for Apple Silicon (Gate 2).

Weight-only ``mx.quantized_matmul`` dequantizes to fp16 and runs fp16
arithmetic — the integer matrix units stay idle, so INT8 buys memory, not
speed (M5 survey). This module is the missing piece: **int8×int8 → int32
accumulate → scale back to float**, implemented as a custom
``mx.fast.metal_kernel``.

Scope of this prototype (local Mac, including M2):

* **Correctness gate** — bit-close to the dequantized reference (the
  quantisation error, not the kernel error).
* **API shape** matching the Gate-1 calibration plan: per-token (per-row)
  activation scales + per-out-channel weight scales.
* **Microbench** vs fp16 ``x @ w.T`` and vs weight-only ``mx.quantized_matmul``.

The **speed gate still needs an M5** (Neural Accelerator integer path). On
M2/M4 this kernel is a scalar-MAC fallback; anything ≤1.2× fp16 is not worth
shipping. See ``docs/design/w8a8_int8_gemm_metal.md``.

Not yet: group-64 weight packing identical to MLX affine, epilogue fusion
into DiT Linear, or QAT (Gate 3).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from fastvideo.logger import init_logger

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Metal sources
# ---------------------------------------------------------------------------

# Naive: one thread per output element. Clear, correct, baseline for the
# tiled kernel. grid = (N, M, 1).
_W8A8_GEMM_NAIVE_SRC = r"""
    uint n = thread_position_in_grid.x;
    uint m = thread_position_in_grid.y;
    if (m >= a_shape[0] || n >= b_shape[0]) {
        return;
    }

    uint K = a_shape[1];
    const device char* A = (const device char*)a;
    const device char* B = (const device char*)b;

    int32_t acc = 0;
    uint a_row = m * K;
    uint b_row = n * K;
    // 4-way unroll helps the M2 scalar path a bit without changing numerics.
    uint k = 0;
    for (; k + 4 <= K; k += 4) {
        acc += int(A[a_row + k])     * int(B[b_row + k]);
        acc += int(A[a_row + k + 1]) * int(B[b_row + k + 1]);
        acc += int(A[a_row + k + 2]) * int(B[b_row + k + 2]);
        acc += int(A[a_row + k + 3]) * int(B[b_row + k + 3]);
    }
    for (; k < K; k++) {
        acc += int(A[a_row + k]) * int(B[b_row + k]);
    }

    float sa = scale_a[m];
    float sb = scale_b[n];
    out[m * b_shape[0] + n] = float(acc) * sa * sb;
"""

# Tiled: each threadgroup cooperatively loads tiles of A/B into threadgroup
# memory and accumulates a TM×TN output tile. Still scalar MACs (no AMX/ANE
# intrinsics exposed through mx.fast.metal_kernel) — mainly reduces global
# loads. grid = (ceil(N/TN)*tg_x, ceil(M/TM)*tg_y, 1).
_TILE_M = 8
_TILE_N = 8
_TILE_K = 32

_W8A8_GEMM_TILED_SRC = r"""
    // Threadgroup tile dims must match the Python constants.
    constexpr uint TM = 8;
    constexpr uint TN = 8;
    constexpr uint TK = 32;

    uint tg_x = threadgroup_position_in_grid.x;
    uint tg_y = threadgroup_position_in_grid.y;
    uint tx = thread_position_in_threadgroup.x;
    uint ty = thread_position_in_threadgroup.y;

    uint M = a_shape[0];
    uint K = a_shape[1];
    uint N = b_shape[0];

    uint m0 = tg_y * TM;
    uint n0 = tg_x * TN;
    uint m = m0 + ty;
    uint n = n0 + tx;

    const device char* A = (const device char*)a;
    const device char* B = (const device char*)b;

    threadgroup char As[TM * TK];
    threadgroup char Bs[TN * TK];

    int32_t acc = 0;
    for (uint k0 = 0; k0 < K; k0 += TK) {
        // Cooperative load of A tile (TM x TK) and B tile (TN x TK).
        // Linearize the TM*TK / TN*TK elements across the TM*TN threads.
        uint tid = ty * TN + tx;
        uint nthreads = TM * TN;

        for (uint i = tid; i < TM * TK; i += nthreads) {
            uint lm = i / TK;
            uint lk = i % TK;
            uint gm = m0 + lm;
            uint gk = k0 + lk;
            char v = 0;
            if (gm < M && gk < K) {
                v = A[gm * K + gk];
            }
            As[i] = v;
        }
        for (uint i = tid; i < TN * TK; i += nthreads) {
            uint ln = i / TK;
            uint lk = i % TK;
            uint gn = n0 + ln;
            uint gk = k0 + lk;
            char v = 0;
            if (gn < N && gk < K) {
                v = B[gn * K + gk];
            }
            Bs[i] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (m < M && n < N) {
            uint kk_max = min(TK, K - k0);
            for (uint kk = 0; kk < kk_max; kk++) {
                acc += int(As[ty * TK + kk]) * int(Bs[tx * TK + kk]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (m < M && n < N) {
        float sa = scale_a[m];
        float sb = scale_b[n];
        out[m * N + n] = float(acc) * sa * sb;
    }
"""

_kernel_cache: dict[str, Any] = {}


def _get_kernel(kind: str = "naive"):
    import mlx.core as mx

    if kind in _kernel_cache:
        return _kernel_cache[kind]
    if kind == "naive":
        source = _W8A8_GEMM_NAIVE_SRC
        name = "w8a8_gemm_naive_v1"
    elif kind == "tiled":
        source = _W8A8_GEMM_TILED_SRC
        name = "w8a8_gemm_tiled_v1"
    else:
        raise ValueError(f"Unknown kernel kind: {kind}")

    kernel = mx.fast.metal_kernel(
        name=name,
        input_names=["a", "b", "scale_a", "scale_b"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )
    _kernel_cache[kind] = kernel
    return kernel


# ---------------------------------------------------------------------------
# Quant helpers (host-side; match Gate-1 "per-token act + per-channel wt")
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class W8A8Matrix:
    """INT8 matrix plus per-row scales such that ``fp ≈ q.astype(f32) * scale[:,None]``."""

    q: Any  # mx.array int8, shape (rows, cols)
    scale: Any  # mx.array float32, shape (rows,)
    rows: int
    cols: int


def quantize_per_row(
    x: np.ndarray | Any,
    *,
    max_abs: float | None = None,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-row absmax → int8. Returns ``(q_int8, scale_f32)``."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {arr.shape}")
    if max_abs is None:
        absmax = np.max(np.abs(arr), axis=1).astype(np.float32)
    else:
        absmax = np.full((arr.shape[0],), float(max_abs), dtype=np.float32)
    absmax = np.maximum(absmax, eps)
    scale = absmax / 127.0
    q = np.clip(np.rint(arr / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale


def quantize_activations_per_token(x: Any) -> W8A8Matrix:
    """Per-token (row) activation quant — Gate-1 recommended scheme."""
    import mlx.core as mx

    if hasattr(x, "shape") and type(x).__module__.startswith("mlx"):
        x_np = np.array(x.astype(mx.float32))
    else:
        x_np = np.asarray(x, dtype=np.float32)
    # Flatten leading dims into tokens: (..., K) → (T, K)
    k = int(x_np.shape[-1])
    tokens = int(np.prod(x_np.shape[:-1])) if x_np.ndim > 1 else 1
    flat = x_np.reshape(tokens, k)
    q, scale = quantize_per_row(flat)
    return W8A8Matrix(
        q=mx.array(q),
        scale=mx.array(scale.astype(np.float32)),
        rows=tokens,
        cols=k,
    )


def quantize_weights_per_out_channel(w: Any) -> W8A8Matrix:
    """Per-out-channel weight quant for ``y = x @ w.T`` (w shape ``[N, K]``)."""
    import mlx.core as mx

    if hasattr(w, "shape") and type(w).__module__.startswith("mlx"):
        w_np = np.array(w.astype(mx.float32))
    else:
        w_np = np.asarray(w, dtype=np.float32)
    if w_np.ndim != 2:
        raise ValueError(f"weights must be 2D [N,K], got {w_np.shape}")
    q, scale = quantize_per_row(w_np)
    return W8A8Matrix(
        q=mx.array(q),
        scale=mx.array(scale.astype(np.float32)),
        rows=int(w_np.shape[0]),
        cols=int(w_np.shape[1]),
    )


def dequant_reference(a: W8A8Matrix, b: W8A8Matrix) -> np.ndarray:
    """``(a.q * sa) @ (b.q * sb).T`` in float64 — kernel correctness oracle."""
    import mlx.core as mx

    aq = np.array(a.q.astype(mx.float32) if hasattr(a.q, "astype") else a.q).astype(np.float64)
    bq = np.array(b.q.astype(mx.float32) if hasattr(b.q, "astype") else b.q).astype(np.float64)
    sa = np.array(a.scale).astype(np.float64).reshape(-1, 1)
    sb = np.array(b.scale).astype(np.float64).reshape(-1, 1)
    return (aq * sa) @ (bq * sb).T


# ---------------------------------------------------------------------------
# Kernel launch
# ---------------------------------------------------------------------------


def w8a8_matmul(
    a: W8A8Matrix,
    b: W8A8Matrix,
    *,
    kind: str = "naive",
) -> Any:
    """Compute ``y = (a.q⊙sa) @ (b.q⊙sb).T`` via the fused Metal kernel.

    Args:
        a: activations ``[M, K]`` int8 + per-row scales.
        b: weights ``[N, K]`` int8 + per-out-channel scales (NOT transposed
           relative to ``nn.Linear.weight`` layout).
        kind: ``"naive"`` or ``"tiled"``.

    Returns:
        ``mx.array`` float32 of shape ``[M, N]``.
    """
    import mlx.core as mx

    if a.cols != b.cols:
        raise ValueError(f"K mismatch: a.cols={a.cols} b.cols={b.cols}")
    m, n, k = a.rows, b.rows, a.cols
    del k  # used only for the check above; Metal reads shapes from buffers

    kernel = _get_kernel(kind)
    if kind == "naive":
        # 8x8 threadgroups tile the output grid; remainder threads just exit.
        tg = (8, 8, 1)
        grid = ((n + tg[0] - 1) // tg[0] * tg[0], (m + tg[1] - 1) // tg[1] * tg[1], 1)
    else:
        tg = (_TILE_N, _TILE_M, 1)
        grid = (
            ((n + _TILE_N - 1) // _TILE_N) * tg[0],
            ((m + _TILE_M - 1) // _TILE_M) * tg[1],
            1,
        )

    out = kernel(
        inputs=[a.q, b.q, a.scale, b.scale],
        output_shapes=[(m, n)],
        output_dtypes=[mx.float32],
        grid=grid,
        threadgroup=tg,
    )
    return out[0]


def w8a8_linear(
    x: Any,
    weight: Any,
    *,
    bias: Any | None = None,
    kind: str = "naive",
) -> Any:
    """Drop-in-ish Linear: quantize ``x`` per-token, ``weight`` per-out, GEMM.

    ``weight`` is ``[out_features, in_features]`` (torch/MLX Linear layout).
    """
    import mlx.core as mx

    a = quantize_activations_per_token(x)
    b = quantize_weights_per_out_channel(weight)
    y = w8a8_matmul(a, b, kind=kind)
    if bias is not None:
        y = y + bias.astype(mx.float32)
    return y


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchRow:
    label: str
    m: int
    n: int
    k: int
    median_ms: float
    tflops: float
    notes: str = ""


def _median_ms(fn, warmup: int, iters: int) -> float:
    import mlx.core as mx

    for _ in range(warmup):
        y = fn()
        mx.eval(y)
    samples = []
    for _ in range(iters):
        # Clear cache between iters so we measure compute, not allocator reuse
        # tricks — but keep it light on 8 GB machines.
        start = time.perf_counter()
        y = fn()
        mx.eval(y)
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2] * 1000.0


def bench_w8a8(
    *,
    m: int = 512,
    n: int = 512,
    k: int = 512,
    warmup: int = 5,
    iters: int = 20,
    seed: int = 0,
) -> list[BenchRow]:
    """Microbench W8A8 naive/tiled vs fp16 GEMM vs weight-only quant matmul."""
    import mlx.core as mx

    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal((m, k)).astype(np.float32)
    w_np = rng.standard_normal((n, k)).astype(np.float32)

    x = mx.array(x_np).astype(mx.float16)
    w = mx.array(w_np).astype(mx.float16)
    a = quantize_activations_per_token(x_np)
    b = quantize_weights_per_out_channel(w_np)

    flops = 2.0 * m * n * k

    rows: list[BenchRow] = []

    def add(label: str, fn, notes: str = "") -> None:
        ms = _median_ms(fn, warmup=warmup, iters=iters)
        tflops = (flops / (ms * 1e-3)) / 1e12
        rows.append(BenchRow(label=label, m=m, n=n, k=k, median_ms=ms, tflops=tflops, notes=notes))
        logger.info("[w8a8 bench] %s: %.3f ms  %.2f TFLOP/s  %s" % (label, ms, tflops, notes))

    add("fp16_gemm", lambda: x @ w.T, notes="baseline mx matmul")
    add("w8a8_naive", lambda: w8a8_matmul(a, b, kind="naive"))
    add("w8a8_tiled", lambda: w8a8_matmul(a, b, kind="tiled"))

    # Weight-only path: quantize W with mx.quantize, keep X fp16.
    try:
        w_q, scales, biases = mx.quantize(w, bits=8, group_size=64)
        mx.eval(w_q, scales, biases)

        def _qmm():
            return mx.quantized_matmul(
                x,
                w_q,
                scales,
                biases,
                transpose=True,
                group_size=64,
                bits=8,
            )

        add("w8a8_weight_only_qmm", _qmm, notes="mx.quantized_matmul (fp16 arith)")
    except Exception as exc:  # pragma: no cover - optional path
        logger.info("[w8a8 bench] weight-only qmm skipped: %s", exc)

    return rows


def _format_bench(rows: list[BenchRow]) -> str:
    lines = [
        f"{'label':28s} {'M':>5} {'N':>5} {'K':>5} {'ms':>10} {'TFLOP/s':>10}  notes",
        "-" * 90,
    ]
    for r in rows:
        lines.append(
            f"{r.label:28s} {r.m:5d} {r.n:5d} {r.k:5d} {r.median_ms:10.3f} {r.tflops:10.3f}  {r.notes}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """CLI: correctness smoke + optional microbench."""
    import argparse

    import mlx.core as mx

    parser = argparse.ArgumentParser(description="W8A8 fused INT8 GEMM Metal prototype (Gate 2)")
    parser.add_argument("--bench", action="store_true", help="Run microbench vs fp16 / weight-only")
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shapes",
        default="",
        help="Comma-separated M×N×K triples for a sweep, e.g. 256x256x256,512x512x512",
    )
    args = parser.parse_args(argv)

    print(f"mlx metal available: {mx.metal.is_available()}")
    print(f"default device: {mx.default_device()}")

    # Always run a small correctness check first.
    rng = np.random.default_rng(args.seed)
    m, n, k = 64, 96, 128
    x = rng.standard_normal((m, k)).astype(np.float32)
    w = rng.standard_normal((n, k)).astype(np.float32)
    a = quantize_activations_per_token(x)
    b = quantize_weights_per_out_channel(w)
    ref = dequant_reference(a, b)
    for kind in ("naive", "tiled"):
        got = np.array(w8a8_matmul(a, b, kind=kind))
        err = float(np.max(np.abs(got - ref)))
        print(f"correctness [{kind}] max_abs_err vs dequant ref: {err:.3e}")
        if err > 1e-3:
            raise SystemExit(f"FAIL {kind}: err {err}")
    print("correctness: PASS")

    if not args.bench:
        return

    shapes: list[tuple[int, int, int]] = []
    if args.shapes:
        for tok in args.shapes.split(","):
            tok = tok.strip().lower().replace("×", "x")
            parts = tok.split("x")
            if len(parts) != 3:
                raise SystemExit(f"bad shape {tok!r}, expected MxNxK")
            shapes.append((int(parts[0]), int(parts[1]), int(parts[2])))
    else:
        shapes.append((args.m, args.n, args.k))

    print("\n=== microbench (median wall-clock; M2 = scalar baseline, M5 = speed gate) ===")
    all_rows: list[BenchRow] = []
    for m_, n_, k_ in shapes:
        # Keep peak buffers modest on 8 GB machines.
        bytes_est = (m_ * k_ + n_ * k_) * (1 + 4)  # int8 + fp32 scales rough
        if bytes_est > 512 * 1024 * 1024:
            print(f"skip {m_}x{n_}x{k_}: estimated buffers too large for this machine")
            continue
        print(f"\n-- shape {m_}x{n_}x{k_} --")
        rows = bench_w8a8(m=m_, n=n_, k=k_, warmup=args.warmup, iters=args.iters, seed=args.seed)
        print(_format_bench(rows))
        all_rows.extend(rows)

        fp = next(r for r in rows if r.label == "fp16_gemm")
        for r in rows:
            if r.label.startswith("w8a8_") and r.label != "w8a8_weight_only_qmm":
                speedup = fp.median_ms / r.median_ms
                print(f"  {r.label} vs fp16: {speedup:.2f}x  "
                      f"{'(below 1.2x ship bar)' if speedup < 1.2 else ''}")


if __name__ == "__main__":
    main()


__all__ = [
    "W8A8Matrix",
    "bench_w8a8",
    "dequant_reference",
    "quantize_activations_per_token",
    "quantize_per_row",
    "quantize_weights_per_out_channel",
    "w8a8_linear",
    "w8a8_matmul",
]
