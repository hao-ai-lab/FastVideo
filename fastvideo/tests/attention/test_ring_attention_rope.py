# SPDX-License-Identifier: Apache-2.0
"""RoPE-slicing correctness for the Ring Attention / USP (Ring+Ulysses) path.

Ulysses can apply RoPE using the full, unsharded ``freqs_cis`` table because
the SP all-to-all gathers the complete sequence before RoPE runs. Ring
Attention never gathers the sequence — each Ring rank only ever holds its own
contiguous chunk (in the USP hybrid, that chunk is first assembled from
several SP shards by the pre-Ring Ulysses all-to-all) — so
``RingAttention._slice_local_rope`` slices the global RoPE table down
to the Ring rank's token range *before* rotating.
"""

from __future__ import annotations

import os

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29511")

import torch

from fastvideo.attention.ring_attention import RingAttention
from fastvideo.layers.rotary_embedding import _apply_rotary_emb


def _random_rope_tables(global_seq_len: int, head_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    angles = torch.rand(global_seq_len, head_size, generator=generator) * 3.14159
    return torch.cos(angles), torch.sin(angles)


def test_local_rope_slice_matches_full_rope_math() -> None:
    """Rotating a shard with the sliced table must match slicing the full
    rotation. This is the mathematical property Ring Attention's per-rank
    RoPE application relies on; it holds regardless of which rank we are, so
    it is checked directly here (no distributed process group needed) rather
    than through ``_slice_local_rope`` (which reads the rank from the SP
    group — see ``test_slice_local_rope_delegates_to_sp_rank`` below).
    """
    world_size = 4
    batch, global_seq_len, heads, head_size = 2, 32, 3, 8
    local_seq_len = global_seq_len // world_size

    generator = torch.Generator().manual_seed(1)
    q = torch.randn(batch, global_seq_len, heads, head_size, generator=generator)
    k = torch.randn(batch, global_seq_len, heads, head_size, generator=generator)
    cos, sin = _random_rope_tables(global_seq_len, head_size)

    full_q = _apply_rotary_emb(q, cos, sin, is_neox_style=False)
    full_k = _apply_rotary_emb(k, cos, sin, is_neox_style=False)

    for rank in range(world_size):
        start = rank * local_seq_len
        end = start + local_seq_len

        local_cos, local_sin = cos[start:end], sin[start:end]
        rotated_q_local = _apply_rotary_emb(q[:, start:end], local_cos, local_sin, is_neox_style=False)
        rotated_k_local = _apply_rotary_emb(k[:, start:end], local_cos, local_sin, is_neox_style=False)

        torch.testing.assert_close(rotated_q_local, full_q[:, start:end], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(rotated_k_local, full_k[:, start:end], rtol=1e-5, atol=1e-6)


def test_slice_local_rope_delegates_to_ring_rank(distributed_setup) -> None:
    """With SP world size 1 (the ``distributed_setup`` fixture), Ring
    Attention is disabled and the Ring rank is always 0, so
    ``_slice_local_rope`` must return exactly the first ``local_seq_len``
    rows of the table."""
    global_seq_len, head_size, local_seq_len = 16, 8, 16
    cos, sin = _random_rope_tables(global_seq_len, head_size)

    local_cos, local_sin = RingAttention._slice_local_rope((cos, sin), local_seq_len)

    torch.testing.assert_close(local_cos, cos[:local_seq_len])
    torch.testing.assert_close(local_sin, sin[:local_seq_len])


def test_slice_local_rope_rejects_out_of_range(distributed_setup) -> None:
    global_seq_len, head_size = 16, 8
    cos, sin = _random_rope_tables(global_seq_len, head_size)

    try:
        RingAttention._slice_local_rope((cos, sin), local_seq_len=global_seq_len + 1)
    except ValueError as exc:
        assert "shorter than the required" in str(exc)
    else:
        raise AssertionError("Expected ValueError for an out-of-range RoPE slice")
