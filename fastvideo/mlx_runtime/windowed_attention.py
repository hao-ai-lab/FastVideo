# SPDX-License-Identifier: Apache-2.0
"""Chunked non-causal sliding-window self-attention for MLX scaling studies.

This module is intentionally standalone (``mlx.core`` + stdlib only) so it can
be micro-benchmarked without pulling in the DiT / FastVideo stack.

Window policy
-------------
**Symmetric** sliding window (non-causal).  For query index ``i`` the allowed
key indices are:

    sinks:   ``j in [0, sink)``  (always visible to every query, if ``sink > 0``)
    local:   ``j in [max(0, i - half), min(S, i + half + 1))``
             where ``half = window // 2``

so each query sees roughly ``window + 1`` local keys (plus any sinks outside
that range).  This is appropriate for a dense, bidirectional DiT denoise pass.

Implementation note (FLOPs)
---------------------------
A full-size additive attention mask still materialises an ``O(S^2)`` score
matrix inside SDPA and does **not** reduce work.  Instead we tile the sequence
into query blocks and run ``mx.fast.scaled_dot_product_attention`` only against
the union of keys that block needs (local slice ± sinks).  That makes
per-block work ``O(chunk * (window + sink) * D)`` and total work
``O(S * (window + sink) * D)``.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx


def _default_scale(head_dim: int, scale: Optional[float]) -> float:
    if scale is not None:
        return float(scale)
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {head_dim}")
    return head_dim**-0.5


def _validate_qkv(q: mx.array, k: mx.array, v: mx.array) -> tuple[int, int, int, int]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            f"q/k/v must be rank-4 (B, H, S, D); got shapes "
            f"{q.shape}, {k.shape}, {v.shape}"
        )
    b, h, s, d = q.shape
    if k.shape != (b, h, s, d) or v.shape != (b, h, s, d):
        raise ValueError(
            f"q/k/v shapes must match exactly; got q={q.shape}, k={k.shape}, v={v.shape}"
        )
    if s == 0:
        raise ValueError("sequence length S must be > 0")
    if d == 0:
        raise ValueError("head dim D must be > 0")
    return b, h, s, d


def full_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: Optional[float] = None,
) -> mx.array:
    """Full (dense) self-attention via fused SDPA.  Shapes ``(B, H, S, D)``."""
    _, _, _, d = _validate_qkv(q, k, v)
    sc = _default_scale(d, scale)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=sc)


def _concat_kv_slices(
    k: mx.array,
    v: mx.array,
    ranges: list[tuple[int, int]],
) -> tuple[mx.array, mx.array]:
    """Concatenate non-overlapping ``[start, end)`` key/value slices along seq."""
    if not ranges:
        raise ValueError("ranges must be non-empty")
    if len(ranges) == 1:
        s0, e0 = ranges[0]
        return k[:, :, s0:e0, :], v[:, :, s0:e0, :]
    k_parts = [k[:, :, s:e, :] for s, e in ranges]
    v_parts = [v[:, :, s:e, :] for s, e in ranges]
    return mx.concatenate(k_parts, axis=2), mx.concatenate(v_parts, axis=2)


def _key_ranges_for_block(
    qs: int,
    qe: int,
    seq_len: int,
    half: int,
    sink: int,
) -> list[tuple[int, int]]:
    """Return ordered, non-overlapping key ranges for a query block ``[qs, qe)``.

    Symmetric local window over every query in the block, plus global sinks
    ``[0, sink)``.  Overlap is merged into a single contiguous range when
    possible so we avoid double-counting sink tokens.
    """
    local_start = max(0, qs - half)
    # Last query index is ``qe - 1``; its right edge is ``qe - 1 + half + 1 = qe + half``.
    local_end = min(seq_len, qe + half)
    if local_start >= local_end:
        # Degenerate (should not happen for valid qs < qe); fall back to sinks only.
        if sink > 0:
            return [(0, min(sink, seq_len))]
        raise ValueError(f"empty local key range for query block [{qs}, {qe})")

    if sink <= 0:
        return [(local_start, local_end)]

    sink_end = min(sink, seq_len)
    if local_start <= sink_end:
        # Sinks abut or overlap the local window — one contiguous slice from 0.
        return [(0, local_end)]
    # Gap between sinks and local window: two slices, concat at SDPA time.
    return [(0, sink_end), (local_start, local_end)]


def windowed_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    window: int,
    sink: int = 0,
    scale: Optional[float] = None,
    *,
    chunk_size: Optional[int] = None,
) -> mx.array:
    """Block-local symmetric sliding-window attention (chunked SDPA).

    Args:
        q, k, v: ``(B, H, S, D)`` query/key/value tensors.
        window: Symmetric window width in tokens (see module docstring).
            Must be ``>= 1``.  When ``window >= S`` the result matches full
            attention up to floating-point noise (sinks are redundant).
        sink: Number of leading key positions that every query may attend to
            globally.  Must be ``>= 0``.
        scale: Softmax scale; defaults to ``1 / sqrt(D)``.
        chunk_size: Query-block length for the FLOP-reducing tile loop.
            Defaults to ``min(window, 512)`` (clamped to at least 1).

    Returns:
        Attention output with the same shape as ``q``.
    """
    _, _, seq_len, d = _validate_qkv(q, k, v)

    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if sink < 0:
        raise ValueError(f"sink must be >= 0, got {sink}")
    if sink > seq_len:
        raise ValueError(f"sink ({sink}) cannot exceed sequence length ({seq_len})")

    sc = _default_scale(d, scale)
    half = window // 2

    # When the requested window is at least the sequence length, every query can
    # see every key under a symmetric policy — fall back to one dense SDPA.
    # (Sinks are redundant once the full key set is used.)
    if window >= seq_len:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=sc)

    if chunk_size is None:
        chunk = min(window, 512)
    else:
        chunk = int(chunk_size)
    if chunk < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk}")
    chunk = min(chunk, seq_len)

    outputs: list[mx.array] = []
    for qs in range(0, seq_len, chunk):
        qe = min(seq_len, qs + chunk)
        q_block = q[:, :, qs:qe, :]
        ranges = _key_ranges_for_block(qs, qe, seq_len, half, sink)
        k_block, v_block = _concat_kv_slices(k, v, ranges)
        if k_block.shape[2] == 0:
            raise RuntimeError(
                f"empty key set for query block [{qs}, {qe}) with window={window}, sink={sink}"
            )
        out_block = mx.fast.scaled_dot_product_attention(
            q_block, k_block, v_block, scale=sc
        )
        outputs.append(out_block)

    return mx.concatenate(outputs, axis=2)
