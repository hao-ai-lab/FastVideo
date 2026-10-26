# SPDX-License-Identifier: Apache-2.0
"""MLX MiniMax H3 VSA tile 128 and the ``p2_multi_region`` Ref2VA policy.

Selection and geometry are compared against the CUDA backend's CPU metadata
builder and ``_build_block_mask``; attention output is compared against a torch
oracle that applies that mask. No GPU is required.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("mlx.core", reason="MLX is required for MiniMax H3 VSA tests")
torch = pytest.importorskip("torch", reason="PyTorch CPU supplies the H3 VSA routing oracle")

import mlx.core as mx  # noqa: E402

from fastvideo.attention.backends import video_sparse_attn_h3 as cuda_vsa  # noqa: E402
from fastvideo.mlx_runtime.minimax_h3 import build_ref2va_packed_layout  # noqa: E402
from fastvideo.mlx_runtime.minimax_h3_reference import MiniMaxH3PreparedReference  # noqa: E402
from fastvideo.mlx_runtime.minimax_h3_vsa import (  # noqa: E402
    MiniMaxH3VSAConfig, VSA_H3_TILE_SHAPES, _block_indices_from_scores, _selected_keep, build_block_mask,
    build_h3_segment_geometry, build_h3_tile_geometry, h3_vsa_attention, ref2va_vsa_segments_from_layout,
    region_sparsities)
from fastvideo.mlx_runtime.minimax_h3_vsa_contract import resolve_vsa_settings  # noqa: E402

_PATCH = (1, 2, 2)
# (raw latent) reference video regions are in latent units; the MLX builder takes token grids.
_REF_RAW = (4, 8, 16)
_TARGET_RAW = (8, 12, 16)
# text, image reference rows, reference video, reference audio rows, target audio, target video
_SEGMENTS_RAW = (9, 6, _REF_RAW, 5, 7, _TARGET_RAW)


def _tokens(raw):
    return tuple(a // p for a, p in zip(raw, _PATCH, strict=True))


def _mlx_segments(raw_segments=_SEGMENTS_RAW):
    return tuple(_tokens(s) if isinstance(s, tuple) else s for s in raw_segments)


def _cuda_meta(raw_segments, tile_size, sparsity, keep_rate):
    return cuda_vsa.MiniMaxH3VSAMetadataBuilder().build(
        current_timestep=0,
        patch_size=_PATCH,
        VSA_sparsity=sparsity,
        packed_segments=raw_segments,
        device=torch.device("cpu"),
        tile_size=tile_size,
        ref_keep_rate=keep_rate,
    )


def test_tile_128_shape_matches_cuda() -> None:
    assert VSA_H3_TILE_SHAPES[128] == cuda_vsa.VSA_H3_TILE_SHAPES[128] == (4, 4, 8)
    assert dict(VSA_H3_TILE_SHAPES) == dict(cuda_vsa.VSA_H3_TILE_SHAPES)


@pytest.mark.parametrize("tile_size", [64, 128, 256])
@pytest.mark.parametrize("raw_segments", [
    _SEGMENTS_RAW,
    (9, 6, 5, 7, _TARGET_RAW),
    (300, (2, 8, 8), 130, (3, 10, 12)),
])
def test_segment_geometry_matches_cuda(raw_segments, tile_size) -> None:
    keep = 0.25 if any(isinstance(s, tuple) for s in raw_segments[:-1]) else None
    meta = _cuda_meta(raw_segments, tile_size, 0.9, keep)
    geo = build_h3_segment_geometry(_mlx_segments(raw_segments), tile_size)
    assert geo.tile_elems == math.prod(VSA_H3_TILE_SHAPES[tile_size])
    assert geo.num_prefix_tiles == meta.num_prefix_tiles
    assert geo.num_video_tiles == meta.num_video_tiles
    assert geo.video_tile_spans == meta.video_tile_spans
    np.testing.assert_array_equal(geo.variable_block_sizes, meta.variable_block_sizes.numpy())
    np.testing.assert_array_equal(geo.untile_combined_index, meta.untile_combined_index.numpy())
    assert len(geo.reference_regions) == meta.reference_video_regions


def test_single_region_wrapper_is_unchanged() -> None:
    geo = build_h3_tile_geometry((7, 5, 3), (8, 4, 6), 128)
    assert geo.video_tile_spans == ((geo.num_prefix_tiles, geo.num_tiles), )
    assert geo.dense_rows_lead
    assert geo.reference_regions == ()


@pytest.mark.parametrize("keep", [0.1, 0.25, 0.3, 0.07, 0.5, 1.0])
@pytest.mark.parametrize("sparsity", [0.9, 0.5])
def test_per_region_topk_budget_matches_cuda(keep, sparsity) -> None:
    """Rounding edge cases: 1 - (1 - 0.7) * n style products must ceil identically."""
    meta = _cuda_meta(_SEGMENTS_RAW, 128, sparsity, keep)
    geo = build_h3_segment_geometry(_mlx_segments(), 128)
    n = geo.num_tiles
    rng = np.random.default_rng(0)
    scores = rng.standard_normal((3, n, n)).astype(np.float32)
    expected = cuda_vsa._build_block_mask(torch.from_numpy(scores)[None], meta.num_prefix_tiles, sparsity, True,
                                          meta.video_tile_spans, meta.span_sparsities)[0].numpy()
    spar = region_sparsities(len(geo.video_tile_spans), sparsity, keep)
    assert spar == pytest.approx(meta.span_sparsities, abs=0, rel=0)
    mask = build_block_mask(scores, geo.num_prefix_tiles, geo.num_video_tiles, sparsity, True, geo.video_tile_spans,
                            spar)
    np.testing.assert_array_equal(mask, expected)
    # The gather path selects the same sets with one constant k per query.
    idx, counts = _block_indices_from_scores(mx.array(scores), geo.num_prefix_tiles, geo.num_video_tiles, sparsity,
                                             True, geo.video_tile_spans, spar)
    idx_np = np.array(idx)
    assert idx_np.shape[-1] == _selected_keep(sparsity, geo.num_prefix_tiles, geo.num_video_tiles, True,
                                              geo.video_tile_spans, spar)
    rebuilt = np.zeros_like(expected[:, geo.num_prefix_tiles:])
    np.put_along_axis(rebuilt, idx_np, True, axis=-1)
    np.testing.assert_array_equal(rebuilt, expected[:, geo.num_prefix_tiles:])
    assert (np.array(counts) == idx_np.shape[-1]).all()


def test_keep_rate_rounding_edge_matches_cuda_float_behaviour() -> None:
    """keep=0.3 over 10 tiles: (1 - (1 - 0.3)) * 10 > 3 in float, so CUDA keeps 4. MLX must too."""
    assert cuda_vsa.compute_topk(1.0 - 0.3, 10) == 4
    from fastvideo.mlx_runtime.minimax_h3_vsa import compute_topk
    assert compute_topk(1.0 - 0.3, 10) == 4
    assert compute_topk(1.0 - 0.1, 30) == cuda_vsa.compute_topk(1.0 - 0.1, 30)


def test_dense_step_keeps_every_region_dense() -> None:
    assert region_sparsities(3, 0.0, 0.1) == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="ref_keep_rate"):
        region_sparsities(2, 0.9, None)


def test_compete_rejects_reference_regions() -> None:
    geo = build_h3_segment_geometry(_mlx_segments(), 128)
    scores = np.zeros((1, geo.num_tiles, geo.num_tiles), dtype=np.float32)
    with pytest.raises(ValueError, match="compete"):
        build_block_mask(scores, geo.num_prefix_tiles, geo.num_video_tiles, 0.9, False, geo.video_tile_spans,
                         region_sparsities(2, 0.9, 0.1))


def _torch_oracle(q, k, v, raw_segments, tile_size, sparsity, keep):
    """Masked attention using the CUDA metadata, mask builder and pooling, on fp32 CPU tensors."""
    meta = _cuda_meta(raw_segments, tile_size, sparsity, keep)
    tile_elems = meta.tile_elems
    seq, heads, dim = q.shape
    n_pad = meta.variable_block_sizes.numel() * tile_elems

    def tile(x):
        buf = torch.zeros(n_pad, heads, dim)
        buf[meta.untile_combined_index] = x
        return buf

    qt, kt, vt = tile(q), tile(k), tile(v)
    qp = cuda_vsa._pool_tiles(qt[None], meta.variable_block_sizes, tile_elems)
    kp = cuda_vsa._pool_tiles(kt[None], meta.variable_block_sizes, tile_elems)
    scores = qp @ kp.transpose(-1, -2) / dim**0.5
    block = cuda_vsa._build_block_mask(scores, meta.num_prefix_tiles, sparsity, True, meta.video_tile_spans,
                                       meta.span_sparsities)
    token_tile, token_valid = cuda_vsa.token_tile_and_valid(meta.variable_block_sizes, tile_elems)
    allow = block[0][:, token_tile][:, :, token_tile] & token_valid[None, None, :]
    out = torch.nn.functional.scaled_dot_product_attention(qt.transpose(0, 1)[None], kt.transpose(0, 1)[None],
                                                           vt.transpose(0, 1)[None], attn_mask=allow[None])
    return out[0].transpose(0, 1)[meta.untile_combined_index]


@pytest.mark.parametrize("tile_size", [64, 128])
@pytest.mark.parametrize("force_gather", [False, True])
def test_multi_region_attention_matches_torch_oracle(tile_size, force_gather, monkeypatch) -> None:
    from fastvideo.mlx_runtime import minimax_h3_vsa as vsa_mod

    if force_gather:
        monkeypatch.setattr(vsa_mod, "_REFERENCE_FULL_MASK_TILE_LIMIT", 0)
    geo = build_h3_segment_geometry(_mlx_segments(), tile_size)
    assert len(geo.video_tile_spans) == 2
    rng = np.random.default_rng(5)
    shape = (geo.total_seq_length, 2, 16)
    q, k, v = (rng.standard_normal(shape).astype(np.float32) for _ in range(3))
    got = h3_vsa_attention(mx.array(q), mx.array(k), mx.array(v), geo, sparsity=0.75, exempt=True, impl="reference",
                           ref_keep_rate=0.25)
    want = _torch_oracle(torch.from_numpy(q), torch.from_numpy(k), torch.from_numpy(v), _SEGMENTS_RAW, tile_size, 0.75,
                         0.25)
    np.testing.assert_allclose(np.array(got), want.numpy(), atol=2e-4, rtol=2e-4)
    # Reference-video queries really are sparse: they differ from the dense result.
    dense = h3_vsa_attention(mx.array(q), mx.array(k), mx.array(v), geo, sparsity=0.0, impl="reference")
    start = 9 + 6  # packed rows of the reference video: (4, 4, 8) tokens = 128
    assert not np.allclose(np.array(got)[start:start + 128], np.array(dense)[start:start + 128], atol=1e-4)


def test_image_only_request_keeps_references_dense() -> None:
    raw = (9, 6, 5, 7, _TARGET_RAW)
    geo = build_h3_segment_geometry(_mlx_segments(raw), 128)
    legacy = build_h3_tile_geometry((9, 6, 5, 7), _tokens(_TARGET_RAW), 128)
    assert geo.video_tile_spans == legacy.video_tile_spans and geo.reference_regions == ()
    np.testing.assert_array_equal(geo.untile_combined_index, legacy.untile_combined_index)
    rng = np.random.default_rng(1)
    shape = (geo.total_seq_length, 2, 16)
    q, k, v = (rng.standard_normal(shape).astype(np.float32) for _ in range(3))
    got = h3_vsa_attention(mx.array(q), mx.array(k), mx.array(v), geo, sparsity=0.75, impl="reference",
                           ref_keep_rate=0.1)  # ignored: there is no reference video region
    want = _torch_oracle(torch.from_numpy(q), torch.from_numpy(k), torch.from_numpy(v), raw, 128, 0.75, 0.1)
    np.testing.assert_allclose(np.array(got), want.numpy(), atol=2e-4, rtol=2e-4)


def _prepared(media_type, frames=0, height=0, width=0, audio=0):
    ref = MiniMaxH3PreparedReference(media_type=media_type, has_audio=audio > 0)
    ref.num_latent_frames, ref.latent_height, ref.latent_width = frames, height, width
    ref.num_audio_latents = audio
    return ref


def test_ref2va_segments_from_layout_mark_only_reference_videos_sparse() -> None:
    layout = build_ref2va_packed_layout(
        np.full(6, 1, dtype=np.int64),
        [_prepared("image", 1, 4, 4), _prepared("video", 2, 4, 8, audio=3)], 2, 4, 4, 3, _PATCH)
    segments = ref2va_vsa_segments_from_layout(layout, _PATCH)
    # text, image rows, video-reference audio rows, video-reference grid, target audio, target grid
    assert segments == (6, 4, 6, (2, 2, 4), 6, (2, 2, 2))
    geo = build_h3_segment_geometry(segments, 128)
    assert geo.total_seq_length == layout.sequence_length
    assert geo.reference_regions == ((2, 2, 4), )
    assert not geo.dense_rows_lead


def test_ref2va_segments_require_reference_spans() -> None:
    layout = build_ref2va_packed_layout(np.full(2, 1, dtype=np.int64), [_prepared("image", 1, 4, 4)], 2, 4, 4, 2,
                                        _PATCH)
    from dataclasses import replace
    with pytest.raises(ValueError, match="per-reference spans"):
        ref2va_vsa_segments_from_layout(replace(layout, reference_segments=()), _PATCH)


_CONTRACT = {
    "attention_backend": "VIDEO_SPARSE_ATTN_H3",
    "vsa_sparsity": 0.9,
    "vsa_tile_size": 128,
    "vsa_ref_policy": "p2_multi_region",
    "vsa_ref_keep_rate": 0.1,
}


def test_contract_supplies_unset_vsa_settings_and_defaults_on() -> None:
    resolved = resolve_vsa_settings(_CONTRACT)
    assert (resolved.enabled, resolved.sparsity, resolved.tile_size, resolved.ref_policy,
            resolved.ref_keep_rate) == (True, 0.9, 128, "p2_multi_region", 0.1)


def test_contract_opt_out_and_mismatched_tile() -> None:
    assert not resolve_vsa_settings(_CONTRACT, vsa=False).enabled
    assert not resolve_vsa_settings(_CONTRACT, vsa=False, tile_size=64).enabled
    with pytest.raises(ValueError, match="vsa_tile_size=128"):
        resolve_vsa_settings(_CONTRACT, tile_size=64)
    overridden = resolve_vsa_settings(_CONTRACT, sparsity=0.8, ref_keep_rate=0.2)
    assert (overridden.sparsity, overridden.ref_keep_rate) == (0.8, 0.2)


def test_no_contract_keeps_legacy_defaults_off() -> None:
    resolved = resolve_vsa_settings(None)
    assert (resolved.enabled, resolved.sparsity, resolved.tile_size, resolved.ref_policy) == (False, 0.9, 64, None)
    assert resolve_vsa_settings(None, vsa=True, tile_size=256).tile_size == 256


def test_dense_only_checkpoint_does_not_default_on() -> None:
    assert not resolve_vsa_settings(_CONTRACT, checkpoint_capable=False).enabled
    assert resolve_vsa_settings(_CONTRACT, vsa=True, checkpoint_capable=False).enabled


def test_config_validates_ref_policy() -> None:
    assert MiniMaxH3VSAConfig(ref_policy="p2_multi_region", ref_keep_rate=0.1).effective_ref_keep_rate == 0.1
    assert MiniMaxH3VSAConfig().effective_ref_keep_rate is None
    for bad in (dict(ref_policy="other", ref_keep_rate=0.1), dict(ref_policy="p2_multi_region"),
                dict(ref_policy="p2_multi_region", ref_keep_rate=0.0), dict(ref_keep_rate=0.1),
                dict(ref_policy="p2_multi_region", ref_keep_rate=1.5)):
        with pytest.raises(ValueError):
            MiniMaxH3VSAConfig(**bad)
