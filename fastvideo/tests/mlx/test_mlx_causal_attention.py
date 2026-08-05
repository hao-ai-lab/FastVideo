# SPDX-License-Identifier: Apache-2.0
"""Track C Rung 3: parity + KV-cache tests for MLX causal self-attention.

Proves the porting insight the streaming runtime relies on — mask-free cached
decoding of one frame-block at a time equals a single block-causal *masked* pass
over the whole sequence — and checks the rolling eviction / sink-token
bookkeeping against explicit references. Backend-agnostic (Metal or mlx[cpu]).
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="MLX is required for the causal-attention tests")

from fastvideo.mlx_runtime.causal import (  # noqa: E402
    MLXCausalKVCache,
    causal_self_attention_step,
    max_attention_size,
)
from fastvideo.mlx_runtime.fastwan import apply_rotary_emb  # noqa: E402

RNG = np.random.default_rng(2026)


def _rand(*shape: int) -> "mx.array":
    return mx.array(RNG.standard_normal(shape).astype(np.float32))


def _qkv_cos_sin(num_tokens: int, num_heads: int, head_dim: int):
    q = _rand(1, num_tokens, num_heads, head_dim)
    k = _rand(1, num_tokens, num_heads, head_dim)
    v = _rand(1, num_tokens, num_heads, head_dim)
    # Rotary tables for absolute positions 0..num_tokens-1 (rope_dim == head_dim).
    cos = _rand(num_tokens, head_dim)
    sin = _rand(num_tokens, head_dim)
    return q, k, v, cos, sin


def _block_causal_masked_reference(q, k, v, cos, sin, *, chunk_tokens: int, window: int, scale: float):
    """Full-sequence attention with a block-causal (optionally windowed) mask.

    Token ``p`` attends every token strictly before the end of ``p``'s chunk and
    no older than ``window`` tokens — the exact set the cached path accumulates.
    """
    num_tokens = q.shape[1]
    roped_q = apply_rotary_emb(q, cos, sin, is_neox_style=False)
    roped_k = apply_rotary_emb(k, cos, sin, is_neox_style=False)

    ends = np.minimum(((np.arange(num_tokens) // chunk_tokens) + 1) * chunk_tokens, num_tokens)
    q_idx = np.arange(num_tokens)[:, None]
    kv_idx = np.arange(num_tokens)[None, :]
    allowed = ((kv_idx < ends[:, None]) & (kv_idx >= (ends[:, None] - window))) | (q_idx == kv_idx)
    mask = mx.array(np.where(allowed, 0.0, -np.inf).astype(np.float32))

    out = mx.fast.scaled_dot_product_attention(
        roped_q.transpose(0, 2, 1, 3),
        roped_k.transpose(0, 2, 1, 3),
        v.transpose(0, 2, 1, 3),
        scale=scale,
        mask=mask,
    ).transpose(0, 2, 1, 3)
    return out


def _run_cached(q, k, v, cos, sin, *, chunk_tokens, local_attn_size, frame_seqlen, kv_cache_size, sink_tokens=0):
    num_tokens = q.shape[1]
    num_heads, head_dim = q.shape[2], q.shape[3]
    cache = MLXCausalKVCache.allocate(
        batch=1, max_tokens=kv_cache_size, num_heads=num_heads, head_dim=head_dim,
        sink_tokens=sink_tokens, dtype=mx.float32)
    outputs = []
    for start in range(0, num_tokens, chunk_tokens):
        end = start + chunk_tokens
        out = causal_self_attention_step(
            q[:, start:end], k[:, start:end], v[:, start:end],
            cos[start:end], sin[start:end], cache,
            current_start=start, local_attn_size=local_attn_size, frame_seqlen=frame_seqlen)
        outputs.append(out)
    return mx.concatenate(outputs, axis=1), cache


def test_cached_matches_block_causal_masked_no_eviction() -> None:
    """local_attn_size=-1: chunked cached decode == block-causal masked full pass."""
    frame_seqlen, nfb, num_frames, num_heads, head_dim = 4, 2, 3, 2, 8
    chunk = frame_seqlen * nfb
    n = num_frames * chunk
    q, k, v, cos, sin = _qkv_cos_sin(n, num_heads, head_dim)
    scale = head_dim**-0.5

    cached, _ = _run_cached(q, k, v, cos, sin, chunk_tokens=chunk, local_attn_size=-1,
                            frame_seqlen=frame_seqlen, kv_cache_size=n)
    ref = _block_causal_masked_reference(q, k, v, cos, sin, chunk_tokens=chunk,
                                         window=max_attention_size(-1, frame_seqlen), scale=scale)
    mx.eval(cached, ref)
    np.testing.assert_allclose(np.array(cached), np.array(ref), atol=2e-4, rtol=2e-4)


def test_cached_matches_sliding_window_with_eviction() -> None:
    """Limited local_attn_size: rolling-cache decode == sliding-window masked pass."""
    frame_seqlen, local_attn_size, num_frames, num_heads, head_dim = 4, 2, 5, 2, 8
    chunk = frame_seqlen  # one frame per block
    n = num_frames * chunk
    window = local_attn_size * frame_seqlen  # 8; cache holds exactly the window
    q, k, v, cos, sin = _qkv_cos_sin(n, num_heads, head_dim)
    scale = head_dim**-0.5

    cached, cache = _run_cached(q, k, v, cos, sin, chunk_tokens=chunk, local_attn_size=local_attn_size,
                                frame_seqlen=frame_seqlen, kv_cache_size=window)
    ref = _block_causal_masked_reference(q, k, v, cos, sin, chunk_tokens=chunk, window=window, scale=scale)
    mx.eval(cached, ref)
    np.testing.assert_allclose(np.array(cached), np.array(ref), atol=2e-4, rtol=2e-4)
    # Indices advance to the full sequence / saturate at the window.
    assert cache.global_end_index == n
    assert cache.local_end_index == window


def test_cached_matches_sliding_window_overlapping_eviction() -> None:
    """window > 2*chunk: eviction shifts overlapping source/dest regions.

    The adjacent-window case (window == 2*chunk) only moves non-overlapping
    slices; production windows are larger, so this exercises the overlapping
    shift path that would corrupt K/V if the rolled copy were not materialised.
    """
    frame_seqlen, local_attn_size, num_frames, num_heads, head_dim = 4, 3, 6, 2, 8
    chunk = frame_seqlen  # 4
    n = num_frames * chunk  # 24
    window = local_attn_size * frame_seqlen  # 12 (> 2*chunk, overlapping shift)
    q, k, v, cos, sin = _qkv_cos_sin(n, num_heads, head_dim)
    scale = head_dim**-0.5

    cached, cache = _run_cached(q, k, v, cos, sin, chunk_tokens=chunk, local_attn_size=local_attn_size,
                                frame_seqlen=frame_seqlen, kv_cache_size=window)
    ref = _block_causal_masked_reference(q, k, v, cos, sin, chunk_tokens=chunk, window=window, scale=scale)
    mx.eval(cached, ref)
    np.testing.assert_allclose(np.array(cached), np.array(ref), atol=2e-4, rtol=2e-4)
    assert cache.global_end_index == n
    assert cache.local_end_index == window


def test_chunk_exceeding_cache_capacity_raises() -> None:
    """Reject chunks larger than the non-sink cache capacity (would clobber sinks)."""
    frame_seqlen, num_heads, head_dim, sink_tokens = 4, 1, 8, 2
    # Capacity after sinks is 2; a 4-token chunk cannot fit without overwriting sinks.
    kv_cache_size = sink_tokens + 2
    cache = MLXCausalKVCache.allocate(
        batch=1, max_tokens=kv_cache_size, num_heads=num_heads, head_dim=head_dim,
        sink_tokens=sink_tokens, dtype=mx.float32)
    # Fill past the cache so the next write triggers overflow eviction.
    q0, k0, v0, cos0, sin0 = _qkv_cos_sin(2, num_heads, head_dim)
    causal_self_attention_step(
        q0, k0, v0, cos0, sin0, cache, current_start=0, local_attn_size=2, frame_seqlen=frame_seqlen)
    q1, k1, v1, cos1, sin1 = _qkv_cos_sin(4, num_heads, head_dim)
    with pytest.raises(ValueError, match="exceeds available cache capacity"):
        causal_self_attention_step(
            q1, k1, v1, cos1, sin1, cache, current_start=2, local_attn_size=2, frame_seqlen=frame_seqlen)


def test_sink_tokens_preserved_across_eviction() -> None:
    """The first ``sink_tokens`` cache slots survive rolling eviction unchanged."""
    frame_seqlen, local_attn_size, num_frames, num_heads, head_dim = 4, 3, 6, 1, 8
    chunk = frame_seqlen
    n = num_frames * chunk
    window = local_attn_size * frame_seqlen  # 12
    sink_tokens = frame_seqlen  # keep the first frame as attention sinks
    q, k, v, cos, sin = _qkv_cos_sin(n, num_heads, head_dim)

    cache = MLXCausalKVCache.allocate(batch=1, max_tokens=window, num_heads=num_heads, head_dim=head_dim,
                                      sink_tokens=sink_tokens, dtype=mx.float32)
    sink_k_after_first = None
    sink_v_after_first = None
    for i, start in enumerate(range(0, n, chunk)):
        end = start + chunk
        causal_self_attention_step(
            q[:, start:end], k[:, start:end], v[:, start:end], cos[start:end], sin[start:end], cache,
            current_start=start, local_attn_size=local_attn_size, frame_seqlen=frame_seqlen)
        if i == 0:
            sink_k_after_first = np.array(cache.k[:, :sink_tokens])
            sink_v_after_first = np.array(cache.v[:, :sink_tokens])
    mx.eval(cache.k, cache.v)
    # After many chunks (and at least one eviction), the sink region is untouched.
    assert cache.global_end_index == n
    np.testing.assert_array_equal(np.array(cache.k[:, :sink_tokens]), sink_k_after_first)
    np.testing.assert_array_equal(np.array(cache.v[:, :sink_tokens]), sink_v_after_first)
