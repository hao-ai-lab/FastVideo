# SPDX-License-Identifier: Apache-2.0
"""CPU checks for VSA-H3 reference-video regions (Ref2VA policy P2) and 128-token tiles.

Under P2 every reference video is its own sparse region, tiled in place at its
packed offset, while text, audio, and image references stay dense. Each
video query keeps its own top-k of every region: the reference keep rate may
differ from the target's. The kernel route for 128-token tiles is exercised
through a stand-in for the sm_100a extension; real kernels are not needed.
"""

import math

import pytest
import torch
import torch.nn.functional as F

import fastvideo.attention.backends.video_sparse_attn_h3 as vsa_h3
import fastvideo.envs as envs
from fastvideo.attention.backends.video_sparse_attn import compute_topk
from fastvideo.attention.backends.video_sparse_attn_h3 import (_TILE_ELEMS, MiniMaxH3VSAImpl,
                                                               MiniMaxH3VSAMetadataBuilder, _build_block_mask,
                                                               _h3_segment_tile_geometry, _h3_tile_geometry,
                                                               _pool_tiles, assert_ref2va_vsa_metadata,
                                                               token_tile_and_valid)
from fastvideo.pipelines.basic.minimax_h3.packing import MINIMAX_H3_TEXT_TAG, build_ref2va_packed_sequence
from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3PreparedReference
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_denoising import _h3_vsa_ref2va_segments

_CPU = torch.device("cpu")
_PATCH = (1, 2, 2)

# packed order: [text 37 | ref_audio 9 | ref_video (5,4,6)=120 | tgt_audio 11
# | tgt_video (8,4,6)=192] -> S=369; raw latents under patch (1,2,2)
_R2V = dict(
    prefix_segments=(37, 9, 11),
    video_segments=((5, 8, 12), (8, 8, 12)),
    video_offsets=(46, 177),
)
_R2V_SEQ = 369
_R2V_DENSE_RUNS = ((0, 37), (37, 46), (166, 177))  # packed [start, end) of text/ref_audio/tgt_audio


def _build_r2v(sparsity=0.0, tile_size=_TILE_ELEMS, device=_CPU, **overrides):
    call = dict(_R2V, **overrides)
    return MiniMaxH3VSAMetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=call.pop("raw_latent_shape", None),
        patch_size=_PATCH,
        VSA_sparsity=sparsity,
        device=device,
        tile_size=tile_size,
        **call,
    )


def _impl(head_size=8):
    return MiniMaxH3VSAImpl(num_heads=2, head_size=head_size, causal=False, softmax_scale=head_size**-0.5)


def _mask(meta, scores, sparsity):
    return _build_block_mask(scores, meta.num_prefix_tiles, meta.num_video_tiles, sparsity, meta.exempt,
                             meta.video_tile_spans, meta.span_sparsities)


def _token_allow(meta, mask, b=0, h=0):
    """Block mask mapped to packed-token coordinates: allow[i, j]."""
    tile_of = meta.untile_combined_index // meta.tile_elems
    return mask[b, h][tile_of][:, tile_of]


def _sparse_attention_oracle(query, key, value, mask, meta):
    """SDPA over the padded tile buffer with the block mask expanded to tokens."""
    token_tile, token_valid = token_tile_and_valid(meta.variable_block_sizes, meta.tile_elems)
    out = torch.empty_like(query)
    for b in range(query.shape[0]):
        for h in range(query.shape[2]):
            allow = mask[b, h][token_tile][:, token_tile] & token_valid[None, :]
            bias = torch.zeros(allow.shape, dtype=query.dtype, device=query.device)
            bias.masked_fill_(~allow, float("-inf"))
            out[b, :, h] = F.scaled_dot_product_attention(query[b, :, h][None], key[b, :, h][None],
                                                          value[b, :, h][None], attn_mask=bias[None])[0]
    return out


# ---------------------------------------------------------------------------
# Packed-order segments for representative Ref2VA layouts
# ---------------------------------------------------------------------------


def _reference(kind, *, frames=1, height=4, width=6, audio=0):
    return MiniMaxH3PreparedReference(media_type=kind,
                                      has_audio=audio > 0,
                                      num_latent_frames=frames,
                                      latent_height=height,
                                      latent_width=width,
                                      num_audio_latents=audio)


def _layout(references, *, text=7, target=(3, 4, 6), audio=5):
    tags = torch.full((text, ), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
    return build_ref2va_packed_sequence(tags, references, *target, audio, _PATCH)


# (references, expected prefix segments, expected video segments, expected offsets).
# Rows: an image (1, 4, 6) latent is 1*2*3 = 6 rows; a (5, 4, 6) video is 30; the
# (3, 4, 6) target is 18; audio rows are 2 per latent; text is 7.
_LAYOUTS = {
    "images_only": ([_reference("image"), _reference("image", height=8, width=4)], (7, 6, 8, 10), ((3, 4, 6), ),
                    (31, )),
    "image_and_silent_video": ([_reference("image"), _reference("video", frames=5)], (7, 6, 10), ((5, 4, 6),
                                                                                                  (3, 4, 6)),
                               (13, 53)),
    "video_with_audio_and_audio": ([_reference("video", frames=5, audio=4),
                                    _reference("audio", audio=3)], (7, 8, 6, 10), ((5, 4, 6), (3, 4, 6)), (15, 61)),
    "two_videos": ([_reference("video", frames=2, height=8, width=4),
                    _reference("video", frames=5, audio=2)], (7, 4, 10), ((2, 8, 4), (5, 4, 6), (3, 4, 6)),
                   (7, 27, 67)),
}


@pytest.mark.parametrize("name", sorted(_LAYOUTS))
@pytest.mark.parametrize("tile_size", [64, 128, 256])
def test_ref2va_segments_tile_every_layout_in_packed_order(name, tile_size):
    references, prefix, videos, offsets = _LAYOUTS[name]
    layout = _layout(references)
    assert _h3_vsa_ref2va_segments(layout, _PATCH) == (prefix, videos, offsets)

    meta = MiniMaxH3VSAMetadataBuilder().build(current_timestep=0,
                                               raw_latent_shape=None,
                                               patch_size=_PATCH,
                                               VSA_sparsity=0.9,
                                               prefix_segments=prefix,
                                               device=_CPU,
                                               tile_size=tile_size,
                                               video_segments=videos,
                                               video_offsets=offsets,
                                               ref_keep_rate=0.1)
    assert meta.total_seq_length == layout.sequence_length
    assert meta.ref2va_policy == "p2_multi_region"
    assert meta.reference_video_regions == len(videos) - 1
    assert_ref2va_vsa_metadata(meta,
                               expected_reference_video_regions=len(videos) - 1,
                               target_sparsity=0.9,
                               ref_keep_rate=0.1)
    # Dense rows (text, audio, image references) map to prefix tiles only;
    # each video region's rows map to its own span.
    tile_of = meta.untile_combined_index // meta.tile_elems
    region_rows = torch.zeros(layout.sequence_length, dtype=torch.bool)
    for (t, h, w), offset, (start, end) in zip(videos, offsets, meta.video_tile_spans, strict=True):
        rows = slice(offset, offset + (t // _PATCH[0]) * (h // _PATCH[1]) * (w // _PATCH[2]))
        region_rows[rows] = True
        assert bool(((tile_of[rows] >= start) & (tile_of[rows] < end)).all())
    assert bool((tile_of[~region_rows] < meta.num_prefix_tiles).all())
    x = torch.randn(1, layout.sequence_length, 2, 4)
    assert torch.equal(_impl().tile(x, meta)[:, meta.untile_combined_index], x)


def test_ref2va_segments_require_reference_spans():
    layout = _layout([_reference("image")])
    stripped = type(layout)(**{**layout.__dict__, "reference_segments": ()})
    with pytest.raises(ValueError, match="per-reference spans"):
        _h3_vsa_ref2va_segments(stripped, _PATCH)


# ---------------------------------------------------------------------------
# Multi-region geometry and per-region top-k
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tile_size,width", [(64, 4), (128, 8)])
def test_multi_region_geometry_native_tile_oracle(tile_size, width):
    """Hand-computed per-region (4,4,4)/(4,4,8) tile ids and sizes at true offsets."""
    meta = _build_r2v(tile_size=tile_size)
    assert meta.total_seq_length == _R2V_SEQ
    assert meta.num_prefix_tiles == 3  # 37, 9, 11 each fit one tile
    per_region = 4 if tile_size == 64 else 2
    assert meta.num_video_tiles == 2 * per_region
    assert meta.video_tile_spans == ((3, 3 + per_region), (3 + per_region, 3 + 2 * per_region))
    assert meta.variable_block_sizes[:3].tolist() == [37, 9, 11]

    idx = meta.untile_combined_index
    for (t, h, w), start, first_tile in (((5, 4, 6), 46, 3), ((8, 4, 6), 177, 3 + per_region)):
        n_t, n_h, n_w = -(-t // 4), -(-h // 4), -(-w // width)
        expected_sizes = torch.tensor([
            min(4, t - 4 * tt) * min(4, h - 4 * hh) * min(width, w - width * ww) for tt in range(n_t)
            for hh in range(n_h) for ww in range(n_w)
        ])
        n_region = n_t * n_h * n_w
        assert torch.equal(meta.variable_block_sizes[first_tile:first_tile + n_region], expected_sizes)
        row = torch.arange(t * h * w)
        row_t, row_h, row_w = row // (h * w), (row // w) % h, row % w
        expected_tile = first_tile + ((row_t // 4) * n_h + row_h // 4) * n_w + row_w // width
        assert torch.equal(idx[start:start + t * h * w] // tile_size, expected_tile)
    # Each prefix tile stays inside ONE dense segment even though the target
    # audio sits between the two video regions.
    for seg_start, seg_end in _R2V_DENSE_RUNS:
        for tile in torch.unique(idx[seg_start:seg_end] // tile_size).tolist():
            assert tile < meta.num_prefix_tiles
            rows = torch.nonzero(idx // tile_size == tile).flatten()
            assert rows.min() >= seg_start and rows.max() < seg_end


def test_single_region_builds_keep_the_legacy_geometry():
    legacy = MiniMaxH3VSAMetadataBuilder().build(current_timestep=0,
                                                 raw_latent_shape=(8, 8, 12),
                                                 patch_size=_PATCH,
                                                 VSA_sparsity=0.5,
                                                 prefix_segments=(37, 9, 120, 11),
                                                 device=_CPU,
                                                 tile_size=128)
    assert legacy.video_tile_spans == () and legacy.span_sparsities == () and legacy.ref2va_policy is None
    general = _h3_segment_tile_geometry((37, 9, 120, 11, (8, 4, 6)), _CPU, (4, 4, 8))
    for actual, expected in zip(_h3_tile_geometry((37, 9, 120, 11), (8, 4, 6), _CPU, (4, 4, 8)), general[:5],
                                strict=True):
        if torch.is_tensor(expected):
            assert torch.equal(actual, expected)
        else:
            assert actual == expected
    assert torch.equal(legacy.untile_combined_index, general[2])


def test_multi_region_span_topk_and_reference_keep_rate():
    """Every video query keeps exactly k_i tiles of EACH region; prefix stays dense."""
    torch.manual_seed(3)
    meta = _build_r2v(sparsity=0.75, tile_size=64)
    P = meta.num_prefix_tiles
    n = P + meta.num_video_tiles
    scores = torch.randn(1, 2, n, n)
    mask = _mask(meta, scores, 0.75)
    assert mask[:, :, :P].all(), "prefix queries dense"
    assert mask[..., :P].all(), "prefix keys visible to every query"
    for (start, end), sparsity in zip(meta.video_tile_spans, meta.span_sparsities, strict=True):
        assert (mask[:, :, P:, start:end].sum(-1) == compute_topk(sparsity, end - start)).all(), (start, end)
    assert meta.span_sparsities == (0.75, 0.75)

    # A span at sparsity 0 keeps every column; the other span keeps its own top-k.
    ref_span, tgt_span = meta.video_tile_spans
    mask = _build_block_mask(scores, P, meta.num_video_tiles, 0.75, meta.exempt, meta.video_tile_spans, (0.0, 0.75))
    assert mask[..., ref_span[0]:ref_span[1]].all(), "a sparsity-0 span's columns are dense"
    assert (mask[:, :, P:, tgt_span[0]:tgt_span[1]].sum(-1) == compute_topk(0.75, tgt_span[1] - tgt_span[0])).all()
    # The builder itself never makes a reference dense: its keep rate is in (0, 1).
    for keep_rate in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match=r"ref_keep_rate must be in \(0, 1\)"):
            _build_r2v(sparsity=0.75, tile_size=64, ref_keep_rate=keep_rate)

    keep_quarter = _build_r2v(sparsity=0.9, tile_size=64, ref_keep_rate=0.25)
    assert keep_quarter.span_sparsities == pytest.approx((0.75, 0.9))
    # A dense build (sparsity 0: dense steps) ignores the reference override.
    assert _build_r2v(sparsity=0.0, ref_keep_rate=0.25).span_sparsities == (0.0, 0.0)
    # Dense layers pass sparsity 0 to the mask builder, which overrides the spans.
    assert _build_block_mask(scores, P, meta.num_video_tiles, 0.0, True, meta.video_tile_spans,
                             meta.span_sparsities).all()


def test_target_only_region_reproduces_the_single_region_token_mask():
    """Folding the reference video back into the prefix must reproduce the legacy policy token for token."""
    torch.manual_seed(4)
    q = torch.randn(1, _R2V_SEQ, 2, 8)
    k = torch.randn(1, _R2V_SEQ, 2, 8)
    builds = (
        dict(prefix_segments=(37, 9, 120, 11), raw_latent_shape=(8, 8, 12), video_segments=None,
             video_offsets=None),
        dict(prefix_segments=(37, 9, 120, 11), video_segments=((8, 8, 12), ), video_offsets=(177, )),
    )
    allows = []
    for build in builds:
        meta = _build_r2v(sparsity=0.75, **build)
        impl = _impl()
        tq, tk = (impl.tile(t, meta).clone() for t in (q, k))
        scores = torch.matmul(_pool_tiles(tq, meta.variable_block_sizes),
                              _pool_tiles(tk, meta.variable_block_sizes).transpose(-2, -1))
        allows.append(torch.stack([_token_allow(meta, _mask(meta, scores, 0.75), 0, h) for h in range(2)]))
    assert torch.equal(allows[0], allows[1])

    # ...and the P2 policy (reference video as its own region) is not that policy.
    meta = _build_r2v(sparsity=0.75)
    impl = _impl()
    tq, tk = (impl.tile(t, meta).clone() for t in (q, k))
    scores = torch.matmul(_pool_tiles(tq, meta.variable_block_sizes),
                          _pool_tiles(tk, meta.variable_block_sizes).transpose(-2, -1))
    p2_allow = torch.stack([_token_allow(meta, _mask(meta, scores, 0.75), 0, h) for h in range(2)])
    assert not torch.equal(allows[0], p2_allow)
    for start, end in _R2V_DENSE_RUNS:
        assert p2_allow[:, start:end].all() and p2_allow[..., start:end].all(), (start, end)
    assert not p2_allow[:, 200, 46:166].all(), "P2 sparsifies reference-video keys"


@pytest.mark.parametrize("tile_size", [_TILE_ELEMS, 64, 128])
def test_multi_region_sparsity_zero_matches_dense_sdpa(tile_size):
    torch.manual_seed(5)
    meta = _build_r2v(tile_size=tile_size)
    q, k, v = (torch.randn(1, _R2V_SEQ, 2, 8) for _ in range(3))
    impl = _impl()
    # An odd 128-token tile count carries one transport-only partner tile; the
    # attention problem is the logical prefix.
    logical = meta.variable_block_sizes.numel() * meta.tile_elems
    tq, tk, tv = (impl.tile(t, meta)[:, :logical].clone() for t in (q, k, v))
    scores = torch.matmul(_pool_tiles(tq, meta.variable_block_sizes, meta.tile_elems),
                          _pool_tiles(tk, meta.variable_block_sizes, meta.tile_elems).transpose(-2, -1))
    mask = _mask(meta, scores, 0.0)
    assert mask.all()
    sparse_out = impl.postprocess_output(_sparse_attention_oracle(tq, tk, tv, mask, meta), meta)
    dense_out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    assert torch.allclose(sparse_out, dense_out, atol=1e-5), (sparse_out - dense_out).abs().max()


def test_multi_region_guard_rejections():
    with pytest.raises(ValueError, match="exactly one of"):
        _build_r2v(raw_latent_shape=(8, 8, 12))
    with pytest.raises(ValueError, match="exactly one of"):
        _build_r2v(video_segments=None, video_offsets=None)
    with pytest.raises(ValueError, match="only to a multi-region"):
        _build_r2v(raw_latent_shape=(8, 8, 12), video_segments=None, ref_keep_rate=0.1)
    with pytest.raises(ValueError, match="one packed-row offset per region"):
        _build_r2v(video_offsets=(46, ))
    with pytest.raises(ValueError, match="do not fill"):
        _build_r2v(video_offsets=(46, 300))
    with pytest.raises(ValueError, match="straddles"):
        _build_r2v(video_offsets=(40, 171))
    with pytest.raises(ValueError, match="packed order"):
        _build_r2v(video_segments=((8, 8, 12), (5, 8, 12)), video_offsets=(177, 46))
    with pytest.raises(ValueError, match="not a positive multiple of patch"):
        _build_r2v(video_segments=((5, 7, 12), (8, 8, 12)))
    for bad in (0.0, 1.5):
        with pytest.raises(ValueError, match="ref_keep_rate"):
            _build_r2v(ref_keep_rate=bad)
    with pytest.raises(ValueError, match="compete"):
        _build_r2v(exempt=False)
    # single-region compete stays supported through the multi-region signature
    meta = _build_r2v(sparsity=0.75, exempt=False, prefix_segments=(37, 20), video_segments=((8, 8, 12), ),
                      video_offsets=(57, ))
    assert meta.exempt is False and len(meta.video_tile_spans) == 1


def test_ref2va_policy_assertion_rejects_p1_and_dense_reference_span():
    legacy_p1 = MiniMaxH3VSAMetadataBuilder().build(current_timestep=0,
                                                    raw_latent_shape=(8, 8, 12),
                                                    patch_size=_PATCH,
                                                    VSA_sparsity=0.9,
                                                    prefix_segments=(37, 9, 120, 11),
                                                    device=_CPU,
                                                    tile_size=64)
    with pytest.raises(ValueError, match="stale single-region layout"):
        assert_ref2va_vsa_metadata(legacy_p1, expected_reference_video_regions=1, target_sparsity=0.9,
                                   ref_keep_rate=0.1)
    with pytest.raises(ValueError, match="leave reference-video conditioning dense"):
        assert_ref2va_vsa_metadata(_build_r2v(sparsity=0.9, tile_size=64, ref_keep_rate=0.1),
                                   expected_reference_video_regions=1,
                                   target_sparsity=0.9,
                                   ref_keep_rate=1.0)
    with pytest.raises(ValueError, match="per-region sparsity policy"):
        assert_ref2va_vsa_metadata(_build_r2v(sparsity=0.9, tile_size=64, ref_keep_rate=0.1),
                                   expected_reference_video_regions=1,
                                   target_sparsity=0.9,
                                   ref_keep_rate=0.2)
    assert_ref2va_vsa_metadata(_build_r2v(sparsity=0.0, tile_size=64, ref_keep_rate=0.1),
                               expected_reference_video_regions=1,
                               target_sparsity=0.0,
                               ref_keep_rate=0.1)


# ---------------------------------------------------------------------------
# 128-token tiles: sm_100a/sm_103a CUDA only
# ---------------------------------------------------------------------------

_HEADS, _DIM = 2, 128


class _FakeSm100a:
    """Stands in for fastvideo_kernel.block_sparse_attn_sm100a."""

    def __init__(self, supported=True):
        self.supported = supported
        self.calls = []

    def is_supported(self, q, variable_block_sizes):
        return self.supported

    def block_sparse_attn_sm100a(self, q, k, v, q2k_idx, q2k_num, variable_block_sizes, need_lse=True):
        self.calls.append(dict(q=q, q2k_num=q2k_num, vbs=variable_block_sizes, need_lse=need_lse))
        return q.clone(), None


def _fake_map_to_index(block_map):
    num = block_map.sum(dim=-1, dtype=torch.int32)
    idx = torch.full(block_map.shape, -1, dtype=torch.int32)
    for position in torch.nonzero(block_map.flatten(0, -2).any(-1)).flatten().tolist():
        cols = torch.nonzero(block_map.flatten(0, -2)[position]).flatten()
        idx.flatten(0, -2)[position, :cols.numel()] = cols.to(torch.int32)
    return idx, num


@pytest.fixture()
def tile128(monkeypatch, env_overrides):
    fake = _FakeSm100a()

    def forbid_triton(*args, **kwargs):
        raise AssertionError("tile 128 must never fall back to the Triton-64 kernel")

    monkeypatch.setattr(vsa_h3, "_sm100a", fake)
    monkeypatch.setattr(vsa_h3, "map_to_index", _fake_map_to_index)
    monkeypatch.setattr(vsa_h3, "block_sparse_attn_64_bhsd", forbid_triton)
    # Tile 128 has no opt-in: it routes to the CUDA kernel with the switch off.
    env_overrides.enter_context(envs.FASTVIDEO_VSA_SM100A.override(None))
    return fake


def _run_tile128(meta, requires_grad=False):
    impl = _impl(_DIM)
    raw = torch.randn(3, meta.total_seq_length, _HEADS, _DIM, dtype=torch.bfloat16, requires_grad=requires_grad)
    tiled = impl.preprocess_qkv(raw, meta)
    query, key, value = tiled.chunk(3, dim=0)
    return tiled, impl.forward(query, key, value, None, meta)


def test_tile128_routes_to_the_cuda_kernel_with_an_odd_tile_partner(tile128):
    meta = _build_r2v(sparsity=0.5, tile_size=128, ref_keep_rate=0.25)
    n_tiles = meta.variable_block_sizes.numel()
    assert n_tiles % 2 == 1
    tiled, out = _run_tile128(meta)
    assert tiled.shape[1] == (n_tiles + 1) * 128
    assert torch.count_nonzero(tiled[:, n_tiles * 128:]) == 0
    assert len(tile128.calls) == 1
    call = tile128.calls[0]
    assert call["need_lse"] is False
    assert call["vbs"].dtype == torch.int32 and call["vbs"].numel() == n_tiles + 1 and call["vbs"][-1] == 0
    assert (call["q2k_num"][..., n_tiles] == 0).all()
    assert out.shape == (1, n_tiles * 128, _HEADS, _DIM)


def test_tile128_even_tile_count_needs_no_partner(tile128):
    meta = _build_r2v(sparsity=0.5, tile_size=128, prefix_segments=(37, 9, 11, 64))
    assert meta.variable_block_sizes.numel() % 2 == 0
    tiled, _ = _run_tile128(meta)
    assert tiled.shape[1] == meta.variable_block_sizes.numel() * 128
    assert len(tile128.calls) == 1


def test_tile128_fails_closed_without_the_cuda_kernel(tile128, monkeypatch):
    meta = _build_r2v(sparsity=0.5, tile_size=128)
    tile128.supported = False
    with pytest.raises(RuntimeError, match="tile 128 requires the sm_100a/sm_103a CUDA block-sparse kernel"):
        _run_tile128(meta)
    monkeypatch.setattr(vsa_h3, "_sm100a", None)
    with pytest.raises(RuntimeError, match="not installed"):
        _run_tile128(meta)


def test_tile128_is_inference_only(tile128):
    meta = _build_r2v(sparsity=0.5, tile_size=128)
    with pytest.raises(NotImplementedError, match="no-grad inference"):
        _run_tile128(meta, requires_grad=True)
    assert tile128.calls == []


def test_tile128_rejects_regional_fullgraph_compile(tile128, monkeypatch):
    meta = _build_r2v(sparsity=0.5, tile_size=128)
    impl = _impl(_DIM)
    impl._regional_compile_sm100a_enabled = True
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    q = torch.zeros(1, meta.variable_block_sizes.numel() * 128, _HEADS, _DIM, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="requires 64-token tiles"):
        impl.forward(q, q, q, None, meta)


def test_tile128_geometry_uses_4x4x8_tiles():
    meta = _build_r2v(tile_size=128)
    assert meta.tile_elems == 128
    assert int(meta.variable_block_sizes.max()) <= 128
    assert int(meta.variable_block_sizes.sum()) == _R2V_SEQ
    # (5, 4, 6) and (8, 4, 6) token grids -> ceil(t/4) * ceil(4/4) * ceil(6/8) tiles each
    assert meta.video_tile_spans == ((3, 5), (5, 7))
    assert math.prod((4, 4, 8)) == 128


@pytest.mark.parametrize("sparsity,ref_keep_rate", [(0.5, 0.25), (0.9, 0.1)])
def test_real_sm100a_tile128_matches_the_token_mask_oracle(env_overrides, sparsity, ref_keep_rate):
    """The real 128-token CUDA forward on a multi-region layout against an FP32 masked-SDPA oracle."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in {(10, 0), (10, 3)}:
        pytest.skip("requires an sm_100a/sm_103a GPU (B200, B300, GB200, GB300)")
    forwards = getattr(vsa_h3._sm100a, "_FWD_BY_BLOCK", {}) if vsa_h3._sm100a is not None else {}
    if forwards.get(128) is None or vsa_h3.map_to_index is None:
        pytest.skip("requires a fastvideo_kernel build with the 128-token sm_100a forward")
    env_overrides.enter_context(envs.FASTVIDEO_VSA_SM100A.override(None))
    device = torch.device("cuda")
    meta = _build_r2v(sparsity=sparsity, tile_size=128, ref_keep_rate=ref_keep_rate, device=device)
    n_tiles = meta.variable_block_sizes.numel()
    logical = n_tiles * 128
    impl = _impl(_DIM)
    torch.manual_seed(19)
    raw = torch.randn(3, meta.total_seq_length, _HEADS, _DIM, device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        tiled = impl.preprocess_qkv(raw, meta)
        assert tiled.shape[1] == (n_tiles + n_tiles % 2) * 128
        query, key, value = tiled.chunk(3, dim=0)
        actual = impl.postprocess_output(impl.forward(query, key, value, None, meta), meta)
        # The mask the backend selected: pooled scores of the logical tiles.
        logical_query, logical_key, logical_value = (t[:, :logical] for t in (query, key, value))
        scores = torch.matmul(_pool_tiles(logical_query, meta.variable_block_sizes, 128),
                              _pool_tiles(logical_key, meta.variable_block_sizes, 128).transpose(-2, -1)) / _DIM**0.5
        mask = _mask(meta, scores, sparsity)
        assert not mask.all(), "the oracle must exercise a sparse mask"
        expected = impl.postprocess_output(
            _sparse_attention_oracle(logical_query.float(), logical_key.float(), logical_value.float(), mask, meta),
            meta)
    torch.cuda.synchronize()
    assert actual.shape == (1, meta.total_seq_length, _HEADS, _DIM)
    torch.testing.assert_close(actual.float(), expected, atol=0.04, rtol=0.02)
