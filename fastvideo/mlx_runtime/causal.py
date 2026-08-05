# SPDX-License-Identifier: Apache-2.0
"""Causal (streaming) self-attention for the MLX FastWan runtime — Track C.

Ports the KV-cached inference path of ``CausalWanSelfAttention`` from
``fastvideo/models/dits/causal_wanvideo.py`` to MLX. The porting insight this
module exists to exploit (see ``docs/design/mac_streaming_causal_guide.md``):

    mask-free cached decoding of one frame-block at a time is identical to a
    single block-causal *masked* pass over the whole sequence.

So the runtime needs no attention mask and no flex-attention — each chunk's
queries attend densely over the cached ``[0:local_end]`` window via
``mx.fast.scaled_dot_product_attention``. The rolling eviction with sink tokens
(which bounds memory for long/streaming rollouts) is reproduced index-for-index
from the torch reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastvideo.mlx_runtime.fastwan import apply_rotary_emb

if TYPE_CHECKING:
    import mlx.core as mx

# When ``local_attn_size == -1`` the torch model keeps a fixed 21-latent-frame
# window for compatibility (``causal_wanvideo.py``); mirror that constant.
GLOBAL_ATTN_COMPAT_MAX_LATENT_FRAMES = 21


@dataclass
class MLXCausalKVCache:
    """Preallocated rolling K/V cache for one attention layer.

    Mirrors the torch ``kv_cache`` dict (``k``, ``v``, ``global_end_index``,
    ``local_end_index``) plus the ``sink_tokens`` count. ``k``/``v`` are
    ``[batch, max_tokens, num_heads, head_dim]``; the *roped* keys are stored
    (rotary is applied before the write, at global positions).
    """

    k: mx.array
    v: mx.array
    global_end_index: int
    local_end_index: int
    sink_tokens: int

    @classmethod
    def allocate(
        cls,
        *,
        batch: int,
        max_tokens: int,
        num_heads: int,
        head_dim: int,
        sink_tokens: int = 0,
        dtype=None,
    ) -> MLXCausalKVCache:
        import mlx.core as mx

        dtype = dtype if dtype is not None else mx.float16
        shape = (batch, max_tokens, num_heads, head_dim)
        return cls(
            k=mx.zeros(shape, dtype=dtype),
            v=mx.zeros(shape, dtype=dtype),
            global_end_index=0,
            local_end_index=0,
            sink_tokens=sink_tokens,
        )


def max_attention_size(local_attn_size: int, frame_seqlen: int) -> int:
    """Attention-window size in tokens, matching the torch reference."""
    if local_attn_size == -1:
        return GLOBAL_ATTN_COMPAT_MAX_LATENT_FRAMES * frame_seqlen
    return local_attn_size * frame_seqlen


def causal_self_attention_step(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    cos: mx.array,
    sin: mx.array,
    cache: MLXCausalKVCache,
    *,
    current_start: int,
    local_attn_size: int,
    frame_seqlen: int,
    scale: float | None = None,
) -> mx.array:
    """One cached, mask-free causal attention step for a frame-block.

    ``q``/``k``/``v`` are ``[batch, num_new_tokens, num_heads, head_dim]`` for
    the current chunk; ``cos``/``sin`` are the rotary tables for this chunk's
    *global* positions (i.e. the caller has already offset by ``current_start``).
    Writes the new roped K/V into ``cache`` (rolling out the oldest tokens past
    the first ``sink_tokens`` on overflow) and returns the chunk's attention
    output ``[batch, num_new_tokens, num_heads, head_dim]``.
    """
    import mlx.core as mx

    num_new = q.shape[1]
    head_dim = q.shape[-1]
    scale = scale if scale is not None else head_dim**-0.5

    roped_query = apply_rotary_emb(q, cos, sin, is_neox_style=False).astype(v.dtype)
    roped_key = apply_rotary_emb(k, cos, sin, is_neox_style=False).astype(v.dtype)

    current_end = current_start + num_new
    sink_tokens = cache.sink_tokens
    window = max_attention_size(local_attn_size, frame_seqlen)
    kv_cache_size = cache.k.shape[1]
    global_end = cache.global_end_index
    local_end_prev = cache.local_end_index

    overflow = (local_attn_size != -1 and current_end > global_end and num_new + local_end_prev > kv_cache_size)
    if overflow:
        # Discard the oldest tokens after the sinks by shifting content left.
        num_evicted = num_new + local_end_prev - kv_cache_size
        num_rolled = local_end_prev - num_evicted - sink_tokens
        # Chunk larger than the non-sink capacity would make num_rolled negative and
        # the subsequent local_start:local_end write would clobber the sink region.
        if num_rolled < 0:
            raise ValueError(
                f"Chunk size ({num_new}) exceeds available cache capacity "
                f"({kv_cache_size - sink_tokens} after sinks); cannot evict "
                f"without overwriting sink tokens.")
        # Copy the source slice first (mx slices are new arrays, so no aliasing).
        rolled_k = cache.k[:, sink_tokens + num_evicted:sink_tokens + num_evicted + num_rolled]
        rolled_v = cache.v[:, sink_tokens + num_evicted:sink_tokens + num_evicted + num_rolled]
        cache.k[:, sink_tokens:sink_tokens + num_rolled] = rolled_k
        cache.v[:, sink_tokens:sink_tokens + num_rolled] = rolled_v
        local_end = local_end_prev + current_end - global_end - num_evicted
    else:
        local_end = local_end_prev + current_end - global_end
    local_start = local_end - num_new

    cache.k[:, local_start:local_end] = roped_key
    cache.v[:, local_start:local_end] = v

    win_start = max(0, local_end - window)
    key_window = cache.k[:, win_start:local_end]
    value_window = cache.v[:, win_start:local_end]

    attn = mx.fast.scaled_dot_product_attention(
        roped_query.transpose(0, 2, 1, 3),
        key_window.transpose(0, 2, 1, 3),
        value_window.transpose(0, 2, 1, 3),
        scale=scale,
    ).transpose(0, 2, 1, 3)

    cache.global_end_index = current_end
    cache.local_end_index = local_end
    return attn
