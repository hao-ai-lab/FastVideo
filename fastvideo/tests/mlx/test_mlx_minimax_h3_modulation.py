# SPDX-License-Identifier: Apache-2.0
"""Exactness gates for compiled MiniMax-H3 AdaLN modulation."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core", reason="MLX is required for MiniMax-H3 modulation tests")

import fastvideo.mlx_runtime.minimax_h3 as h3  # noqa: E402


@pytest.mark.parametrize("hidden_dtype", [mx.float32, mx.bfloat16, mx.float16])
@pytest.mark.parametrize("table_dtype", [mx.float32, mx.bfloat16, mx.float16])
@pytest.mark.parametrize("strided", [False, True])
def test_compiled_modulation_matches_eager_for_supported_dtypes(hidden_dtype, table_dtype, strided) -> None:
    rows = 17
    mx.random.seed(2026)
    hidden = mx.random.normal((rows * 2 if strided else rows, 128), dtype=hidden_dtype)
    if strided:
        hidden = hidden[::2]
    scale = mx.random.normal((9, 128), dtype=table_dtype)
    shift = mx.random.normal((9, 128), dtype=table_dtype)
    indices = mx.arange(rows) % 9

    expected = hidden * (1.0 + scale[indices]) + shift[indices]
    actual = h3._modulate(hidden, (1.0 + scale)[indices], shift[indices])
    mx.eval(expected, actual)

    assert bool(mx.array_equal(actual, expected).item())
