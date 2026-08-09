# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the VSA-H3 backend: tiling geometry, mask policy, and
end-to-end equivalence against dense SDPA through a token-level mask
reference. The same reference doubles as the GPU kernel parity oracle."""

import math

import torch
import torch.nn.functional as F

from fastvideo.attention.backends.video_sparse_attn_h3 import (_TILE_ELEMS, MiniMaxH3VSAImpl,
                                                               MiniMaxH3VSAMetadataBuilder, _build_block_mask,
                                                               _pool_tiles, token_tile_and_valid)

_720P = dict(raw_latent_shape=(30, 44, 80), patch_size=(1, 2, 2), prefix_segments=(512, 1760, 400))
_TINY = dict(raw_latent_shape=(8, 8, 12), patch_size=(1, 2, 2), prefix_segments=(7, 5, 3))

_CPU = torch.device("cpu")


def _build(spec, sparsity=0.0, device=_CPU):
    return MiniMaxH3VSAMetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=spec["raw_latent_shape"],
        patch_size=spec["patch_size"],
        VSA_sparsity=sparsity,
        prefix_segments=spec["prefix_segments"],
        device=device,
    )


def _impl():
    return MiniMaxH3VSAImpl(num_heads=2, head_size=8, causal=False, softmax_scale=8**-0.5)


def reference_sparse_attention(query, key, value, mask, meta):
    """Token-level oracle: SDPA over the padded tile buffer with the block
    mask expanded to tokens. query/key/value: tiled [B, S_pad, H, D]."""
    token_tile, token_valid = token_tile_and_valid(meta.variable_block_sizes)
    out = torch.empty_like(query)
    for b in range(query.shape[0]):
        for h in range(query.shape[2]):
            allow = mask[b, h][token_tile][:, token_tile] & token_valid[None, :]
            bias = torch.zeros(allow.shape, dtype=query.dtype, device=query.device)
            bias.masked_fill_(~allow, float("-inf"))
            out[b, :, h] = F.scaled_dot_product_attention(
                query[b, :, h][None],
                key[b, :, h][None],
                value[b, :, h][None],
                attn_mask=bias[None],
            )[0]
    return out


def test_geometry_720p():
    meta = _build(_720P)
    seq = meta.total_seq_length
    assert seq == 512 + 1760 + 400 + 26400
    assert meta.num_prefix_tiles == 2 + 7 + 2
    assert meta.num_video_tiles == 8 * 3 * 5
    assert int(meta.variable_block_sizes.sum()) == seq
    # (permutation coverage of [0, seq) is implied by the roundtrip below:
    # untile_combined_index scatters seq distinct rows and recovers all of x)
    # segment purity: no prefix tile straddles a segment boundary
    boundaries = [512, 512 + 1760, 512 + 1760 + 400]
    start = 0
    for size in meta.variable_block_sizes[:meta.num_prefix_tiles].tolist():
        end = start + size
        assert all(not (start < b < end) for b in boundaries), (start, end)
        start = end
    # untile(tile(x)) == x
    x = torch.randn(1, seq, 2, 4)
    buf = _impl().tile(x, meta)
    assert buf.shape[1] == meta.variable_block_sizes.numel() * _TILE_ELEMS
    assert torch.equal(buf[:, meta.untile_combined_index], x)


def test_mask_policy():
    meta = _build(_720P, sparsity=0.9)
    n = meta.num_prefix_tiles + meta.num_video_tiles
    P, V = meta.num_prefix_tiles, meta.num_video_tiles
    k_vid = math.ceil(0.1 * V)
    scores = torch.randn(1, 2, n, n)

    exempt = _build_block_mask(scores, P, V, 0.9, exempt=True)
    assert exempt[:, :, :P].all(), "prefix queries must be dense"
    assert exempt[..., :P].all(), "prefix keys must be visible to every query"
    assert (exempt[:, :, P:, P:].sum(-1) == k_vid).all(), "video rows select exactly k_vid video tiles"

    compete = _build_block_mask(scores, P, V, 0.9, exempt=False)
    assert compete[:, :, :P].all()
    assert (compete[:, :, P:].sum(-1) == min(k_vid + P, n)).all(), "budget-matched top-k"

    dense = _build_block_mask(scores, P, V, 0.0, exempt=True)
    assert dense.all(), "sparsity 0 must select everything"


def test_sparsity_zero_matches_dense_sdpa():
    torch.manual_seed(0)
    meta = _build(_TINY)
    seq = meta.total_seq_length
    q, k, v = (torch.randn(1, seq, 2, 8) for _ in range(3))
    impl = _impl()
    tq, tk, tv = (impl.tile(t, meta).clone() for t in (q, k, v))

    scores = torch.matmul(_pool_tiles(tq, meta.variable_block_sizes),
                          _pool_tiles(tk, meta.variable_block_sizes).transpose(-2, -1))
    mask = _build_block_mask(scores, meta.num_prefix_tiles, meta.num_video_tiles, 0.0, exempt=True)
    sparse_out = impl.postprocess_output(reference_sparse_attention(tq, tk, tv, mask, meta), meta)

    dense_out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    assert torch.allclose(sparse_out, dense_out, atol=1e-5), (sparse_out - dense_out).abs().max()


def test_prefix_queries_stay_dense_at_high_sparsity():
    torch.manual_seed(1)
    meta = _build(_TINY, sparsity=0.75)
    seq = meta.total_seq_length
    prefix_len = sum(_TINY["prefix_segments"])
    q, k, v = (torch.randn(1, seq, 2, 8) for _ in range(3))
    impl = _impl()
    tq, tk, tv = (impl.tile(t, meta).clone() for t in (q, k, v))

    scores = torch.matmul(_pool_tiles(tq, meta.variable_block_sizes),
                          _pool_tiles(tk, meta.variable_block_sizes).transpose(-2, -1))
    for exempt in (True, False):
        mask = _build_block_mask(scores, meta.num_prefix_tiles, meta.num_video_tiles, 0.75, exempt=exempt)
        sparse_out = impl.postprocess_output(reference_sparse_attention(tq, tk, tv, mask, meta), meta)
        dense_out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                                   v.transpose(1, 2)).transpose(1, 2)
        assert torch.allclose(sparse_out[:, :prefix_len], dense_out[:, :prefix_len], atol=1e-5)
        assert not torch.allclose(sparse_out[:, prefix_len:], dense_out[:, prefix_len:], atol=1e-5), \
            "video rows should actually be sparse at 75%"


if __name__ == "__main__":
    test_geometry_720p()
    test_mask_policy()
    test_sparsity_zero_matches_dense_sdpa()
    test_prefix_queries_stay_dense_at_high_sparsity()
    print("all VSA-H3 CPU checks passed")
