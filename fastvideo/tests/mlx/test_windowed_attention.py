# SPDX-License-Identifier: Apache-2.0
"""Correctness checks for chunked non-causal windowed attention (MLX)."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="MLX is required for windowed-attention tests")

from fastvideo.mlx_runtime.windowed_attention import (  # noqa: E402
    _key_ranges_for_block,
    full_attention,
    windowed_attention,
)


def _rand_qkv(b: int, h: int, s: int, d: int, seed: int = 0):
    """
    Generate deterministic random query, key, and value tensors.

    Parameters:
        b (int): Batch size.
        h (int): Number of attention heads.
        s (int): Sequence length.
        d (int): Feature dimension.
        seed (int): Random number generator seed.

    Returns:
        tuple: Query, key, and value tensors with shape `(b, h, s, d)`.
    """
    rng = np.random.default_rng(seed)
    q = mx.array(rng.standard_normal((b, h, s, d)).astype(np.float32))
    k = mx.array(rng.standard_normal((b, h, s, d)).astype(np.float32))
    v = mx.array(rng.standard_normal((b, h, s, d)).astype(np.float32))
    return q, k, v


def _mean_cosine(a: "mx.array", b: "mx.array") -> float:
    """
    Compute the mean cosine similarity between corresponding feature vectors.

    Parameters:
        a: The first tensor.
        b: The second tensor.

    Returns:
        The mean cosine similarity, with protection against zero-norm vectors.
    """
    a32 = np.array(a.astype(mx.float32)).reshape(-1, a.shape[-1])
    b32 = np.array(b.astype(mx.float32)).reshape(-1, b.shape[-1])
    num = np.sum(a32 * b32, axis=-1)
    den = np.linalg.norm(a32, axis=-1) * np.linalg.norm(b32, axis=-1)
    return float(np.mean(num / np.maximum(den, 1e-12)))


def test_windowed_matches_full_when_window_ge_seq() -> None:
    """``window >= S`` must recover full attention (cosine > 0.999)."""
    b, h, s, d = 1, 2, 64, 16
    q, k, v = _rand_qkv(b, h, s, d, seed=7)
    scale = d**-0.5

    ref = full_attention(q, k, v, scale=scale)
    out = windowed_attention(q, k, v, window=s, sink=0, scale=scale)
    mx.eval(ref, out)

    assert out.shape == q.shape
    cos = _mean_cosine(ref, out)
    assert cos > 0.999, f"expected cosine > 0.999 for window>=S, got {cos}"


def test_windowed_output_shape_equals_input() -> None:
    b, h, s, d = 2, 3, 48, 8
    q, k, v = _rand_qkv(b, h, s, d, seed=11)
    out = windowed_attention(q, k, v, window=16, sink=4)
    mx.eval(out)
    assert out.shape == q.shape
    assert out.shape == (b, h, s, d)


def test_windowed_with_sinks_shape_and_finite() -> None:
    b, h, s, d = 1, 2, 40, 8
    q, k, v = _rand_qkv(b, h, s, d, seed=19)
    out = windowed_attention(q, k, v, window=8, sink=4, chunk_size=5)
    mx.eval(out)
    assert out.shape == (b, h, s, d)
    arr = np.array(out)
    assert np.isfinite(arr).all()


def test_windowed_result_does_not_depend_on_chunk_size() -> None:
    q, k, v = _rand_qkv(1, 2, 40, 8, seed=23)
    per_query = windowed_attention(q, k, v, window=8, sink=4, chunk_size=1)
    chunked = windowed_attention(q, k, v, window=8, sink=4, chunk_size=5)
    mx.eval(per_query, chunked)

    np.testing.assert_allclose(np.array(chunked), np.array(per_query), rtol=1e-5, atol=1e-5)


def test_sink_range_extends_beyond_local_window() -> None:
    assert _key_ranges_for_block(qs=0, qe=1, seq_len=12, half=1, sink=5) == [(0, 5)]


def test_windowed_attention_enforces_per_query_local_and_sink_mask() -> None:
    q = mx.zeros((1, 1, 6, 1))
    k = mx.zeros_like(q)
    v = mx.arange(6).reshape(1, 1, 6, 1)

    out = windowed_attention(q, k, v, window=2, sink=3, chunk_size=4)
    mx.eval(out)

    np.testing.assert_allclose(np.array(out).reshape(-1), [1.0, 1.0, 1.5, 2.0, 2.5, 2.4], atol=1e-6)


def test_invalid_window_raises() -> None:
    q, k, v = _rand_qkv(1, 1, 8, 4, seed=1)
    with pytest.raises(ValueError, match="window"):
        windowed_attention(q, k, v, window=0)
