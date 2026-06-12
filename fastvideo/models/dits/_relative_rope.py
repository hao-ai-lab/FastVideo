# SPDX-License-Identifier: Apache-2.0
"""Relativistic RoPE re-indexing for causal video DiTs.

Re-maps the KV-cache window to a fixed [0, max_attention_frames) each step (keys
from 0, query at the tail) so positions stay in the trained range as the rollout
grows; needs un-roped keys in the cache.
"""


def relativistic_window_offsets(
    local_end_index: int,
    num_new_tokens: int,
    max_attention_size: int,
) -> tuple[int, int, int]:
    """Token offsets into a position-0 cos/sin table for one re-indexing step.

    Returns ``(window_len, query_lo, query_hi)``: the cached window occupies
    ``table[0:window_len]`` and the query the tail ``table[query_lo:query_hi]``.
    """
    window_len = min(local_end_index, max_attention_size)
    return window_len, window_len - num_new_tokens, window_len
