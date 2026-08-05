# SPDX-License-Identifier: Apache-2.0
"""Accuracy + capability probe for MLX block-scaled quant backends.

For each backend in :mod:`fastvideo.mlx_runtime.quant_backends`, if the installed
MLX build supports the mode, assert that quantized matmul stays within a
per-format relative Frobenius error budget vs. the fp16 reference ``x @ w.T``.
Unsupported modes are skipped with an explicit reason.

Run with ``-s`` to print the summary table (backend | supported | bytes/weight |
rel-error).
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx.core", reason="MLX is required for quant backend tests")

import mlx.core as mx  # noqa: E402

from fastvideo.mlx_runtime.quant_backends import (  # noqa: E402
    BACKENDS,
    bytes_per_weight,
    is_supported,
    quantize_weight,
    quantized_matmul,
    support_error,
)

# Fixed shapes: weight cols must be divisible by all group sizes (64/32/16).
_WEIGHT_ROWS = 1024
_WEIGHT_COLS = 1024
_BATCH = 8
_SEED = 0

# Relative Frobenius error ||y - ref||_F / ||ref||_F tolerances.
# Measured once on mlx 0.31.2 / Apple Silicon for N(0,1) fp16 weights:
#   affine_int8_g64 ~ 5e-3, mxfp8 ~ 8e-2, mxfp4 ~ 1.2e-1, nvfp4 ~ 1.0e-1.
# Budgets keep ~2x headroom for RNG / platform variance.
_REL_ERROR_TOL: dict[str, float] = {
    "affine_int8_g64": 2e-2,
    "mxfp8": 1.5e-1,
    "mxfp4": 2.5e-1,
    "nvfp4": 2.5e-1,
}


def _frobenius_rel_error(y: mx.array, ref: mx.array) -> float:
    """||y - ref||_F / ||ref||_F in float32 (avoids fp16 sum overflow)."""
    y32 = y.astype(mx.float32)
    ref32 = ref.astype(mx.float32)
    diff = y32 - ref32
    num = mx.sqrt(mx.sum(diff * diff))
    den = mx.sqrt(mx.sum(ref32 * ref32)) + mx.array(1e-12, dtype=mx.float32)
    mx.eval(num, den)
    return float(num / den)


def _make_inputs() -> tuple[mx.array, mx.array, mx.array]:
    mx.random.seed(_SEED)
    w = mx.random.normal((_WEIGHT_ROWS, _WEIGHT_COLS)).astype(mx.float16)
    x = mx.random.normal((_BATCH, _WEIGHT_COLS)).astype(mx.float16)
    ref = x @ w.T
    mx.eval(w, x, ref)
    return w, x, ref


@pytest.mark.parametrize("backend", list(BACKENDS))
def test_quant_backend_matmul_accuracy(backend: str) -> None:
    if not is_supported(backend):
        err = support_error(backend) or "unknown error"
        pytest.skip(reason=f"{backend} not supported by installed MLX: {err}")

    w, x, ref = _make_inputs()
    qw = quantize_weight(w, backend)
    y = quantized_matmul(x, qw)
    mx.eval(y)

    rel = _frobenius_rel_error(y, ref)
    tol = _REL_ERROR_TOL[backend]
    assert rel < tol, (
        f"{backend}: relative Frobenius error {rel:.6e} exceeds tolerance {tol:.6e}"
    )


def test_quant_backends_summary_table(capsys: pytest.CaptureFixture[str]) -> None:
    """Print a one-row-per-backend summary when run with pytest -s."""
    w, x, ref = _make_inputs()
    rows: list[tuple[str, str, str, str]] = []

    for backend in BACKENDS:
        supported = is_supported(backend)
        if not supported:
            rows.append((backend, "no", "n/a", "n/a"))
            continue

        bpw = bytes_per_weight(backend)
        qw = quantize_weight(w, backend)
        y = quantized_matmul(x, qw)
        mx.eval(y)
        rel = _frobenius_rel_error(y, ref)
        rows.append((backend, "yes", f"{bpw:.6f}", f"{rel:.6e}"))

    # Header + rows for human inspection under pytest -s.
    col_w = max(len(r[0]) for r in rows)
    header = f"{'backend':<{col_w}}  supported  bytes/weight  rel-error"
    print("\n" + header)
    print("-" * len(header))
    for backend, supported, bpw, rel in rows:
        print(f"{backend:<{col_w}}  {supported:<9}  {bpw:>12}  {rel}")

    # At least the affine baseline must work on any supported MLX install.
    assert is_supported("affine_int8_g64"), (
        "affine_int8_g64 must be supported; cannot evaluate other backends"
    )
