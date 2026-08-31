# SPDX-License-Identifier: Apache-2.0
"""MLX MiniMax H3 VSA: geometry, routing, conversion, and SIMD parity.

None of these tests require a CUDA GPU. Geometry and routing compare against
the CPU PyTorch H3 backend; attention tests use tiny deterministic MLX tensors.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx.core", reason="MLX is required for MiniMax H3 VSA tests")
torch = pytest.importorskip("torch", reason="PyTorch CPU supplies the H3 VSA routing oracle")

import mlx.core as mx  # noqa: E402

from fastvideo.attention.backends.video_sparse_attn_h3 import (  # noqa: E402
    MiniMaxH3VSAMetadataBuilder,
    _build_block_mask,
)
from fastvideo.mlx_runtime.fastwan import MLXQuantizationSpec, quantize_matrix  # noqa: E402
from fastvideo.mlx_runtime.minimax_h3 import (  # noqa: E402
    MiniMaxH3PackedLayout,
    _is_ignored_dense_key,
    _is_quantizable,
    load_mlx_h3_checkpoint,
    mlx_h3_checkpoint_vsa_capable,
    save_mlx_h3_checkpoint,
)
from fastvideo.mlx_runtime.minimax_h3_vsa import (  # noqa: E402
    MiniMaxH3VSAConfig,
    PUBLIC_VSA_IMPLS,
    VSA_GATE_KEY_SUFFIX,
    _untile_hidden,
    build_block_mask,
    build_h3_tile_geometry,
    compute_topk,
    expected_vsa_gate_keys,
    h3_vsa_attention,
    parse_dense_layers,
    prefix_segments_from_layout,
    resolve_impl,
    token_tile_and_valid,
)
from fastvideo.mlx_runtime.minimax_h3_vsa_simd import simd_kernel_available  # noqa: E402

_TINY = dict(raw_latent_shape=(8, 8, 12), patch_size=(1, 2, 2), prefix_segments=(7, 5, 3))
_TINY64 = dict(raw_latent_shape=(9, 20, 26), patch_size=(1, 2, 2), prefix_segments=(70, 5, 130))
_720P = dict(raw_latent_shape=(30, 44, 80), patch_size=(1, 2, 2), prefix_segments=(512, 1760, 400))
_CPU = torch.device("cpu")


def _dit_shape(spec: dict) -> tuple[int, int, int]:
    raw, patch = spec["raw_latent_shape"], spec["patch_size"]
    return raw[0] // patch[0], raw[1] // patch[1], raw[2] // patch[2]


def _torch_meta(spec: dict, tile_size: int = 256, sparsity: float = 0.0):
    return MiniMaxH3VSAMetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=spec["raw_latent_shape"],
        patch_size=spec["patch_size"],
        VSA_sparsity=sparsity,
        prefix_segments=spec["prefix_segments"],
        device=_CPU,
        tile_size=tile_size,
    )


def test_geometry_matches_pytorch_720p_and_packed_order() -> None:
    spec = _720P
    geo = build_h3_tile_geometry(spec["prefix_segments"], _dit_shape(spec), tile_size=256)
    meta = _torch_meta(spec, tile_size=256)
    assert geo.total_seq_length == meta.total_seq_length == 512 + 1760 + 400 + 26400
    assert geo.num_prefix_tiles == int(meta.num_prefix_tiles)
    assert geo.num_video_tiles == int(meta.num_video_tiles)
    assert np.array_equal(geo.variable_block_sizes, meta.variable_block_sizes.numpy())
    assert np.array_equal(geo.untile_combined_index, meta.untile_combined_index.numpy())
    boundaries = [512, 512 + 1760, 512 + 1760 + 400]
    start = 0
    for size in geo.variable_block_sizes[:geo.num_prefix_tiles].tolist():
        end = start + size
        assert all(not (start < boundary < end) for boundary in boundaries)
        start = end
    x = np.arange(geo.total_seq_length, dtype=np.int64)
    buf = np.zeros(geo.padded_length, dtype=np.int64)
    buf[geo.untile_combined_index] = x
    assert np.array_equal(buf[geo.untile_combined_index], x)


def test_geometry_tile64_ragged_tails() -> None:
    spec = _TINY64
    geo = build_h3_tile_geometry(spec["prefix_segments"], _dit_shape(spec), tile_size=64)
    meta = _torch_meta(spec, tile_size=64)
    assert geo.tile_elems == 64
    assert geo.variable_block_sizes[:geo.num_prefix_tiles].tolist() == [64, 6, 5, 64, 64, 2]
    assert np.array_equal(geo.variable_block_sizes, meta.variable_block_sizes.numpy())
    assert np.array_equal(geo.untile_combined_index, meta.untile_combined_index.numpy())


def test_routing_mask_parity_exempt_and_compete() -> None:
    spec = _720P
    geo = build_h3_tile_geometry(spec["prefix_segments"], _dit_shape(spec), 256)
    rng = np.random.default_rng(0)
    n = geo.num_tiles
    scores = rng.standard_normal((1, 2, n, n)).astype(np.float32)
    p, v = geo.num_prefix_tiles, geo.num_video_tiles
    k_vid = compute_topk(0.9, v)
    for exempt in (True, False):
        mlx_mask = build_block_mask(scores, p, v, 0.9, exempt=exempt)
        torch_mask = _build_block_mask(torch.tensor(scores), p, v, 0.9, exempt=exempt).numpy()
        assert mlx_mask[:, :, :p].all()
        if exempt:
            assert mlx_mask[..., :p].all()
            assert (mlx_mask[:, :, p:, p:].sum(-1) == k_vid).all()
        else:
            assert (mlx_mask[:, :, p:].sum(-1) == min(k_vid + p, n)).all()
        assert np.array_equal(mlx_mask, torch_mask)


def test_dense_non_video_queries_and_sparsity_zero_select_all() -> None:
    spec = _TINY
    geo = build_h3_tile_geometry(spec["prefix_segments"], _dit_shape(spec), 256)
    scores = np.random.default_rng(1).standard_normal((1, 2, geo.num_tiles, geo.num_tiles))
    dense = build_block_mask(scores, geo.num_prefix_tiles, geo.num_video_tiles, 0.0, exempt=True)
    assert dense.all()
    sparse = build_block_mask(scores, geo.num_prefix_tiles, geo.num_video_tiles, 0.75, exempt=True)
    assert sparse[:, :, :geo.num_prefix_tiles].all()


def test_pool_tiles_uses_fp32_accumulation() -> None:
    from fastvideo.attention.backends.video_sparse_attn_h3 import _pool_tiles as torch_pool
    from fastvideo.mlx_runtime.minimax_h3_vsa import _pool_tiles, _tile_hidden

    spec = _TINY
    geo = build_h3_tile_geometry(spec["prefix_segments"], _dit_shape(spec), 256)
    mx.random.seed(11)
    packed = mx.random.normal((geo.total_seq_length, 2, 8)).astype(mx.bfloat16)
    tiled = _tile_hidden(packed, geo)
    mlx_pooled = np.array(_pool_tiles(tiled, geo.variable_block_sizes, geo.tile_elems))
    tiled_f32 = np.array(tiled.astype(mx.float32))
    torch_pooled = torch_pool(
        torch.tensor(tiled_f32, dtype=torch.bfloat16)[None],
        torch.tensor(geo.variable_block_sizes, dtype=torch.int64),
        geo.tile_elems,
    )[0].float().numpy()
    np.testing.assert_allclose(mlx_pooled, torch_pooled, rtol=2e-3, atol=2e-3)


def test_block_indices_sets_match_argsort_mask() -> None:
    from fastvideo.mlx_runtime.minimax_h3_vsa import _block_indices_from_scores

    spec = _TINY64
    geo = build_h3_tile_geometry(spec["prefix_segments"], _dit_shape(spec), 64)
    rng = np.random.default_rng(12)
    scores = rng.standard_normal((2, geo.num_tiles, geo.num_tiles)).astype(np.float32)
    mask = build_block_mask(scores, geo.num_prefix_tiles, geo.num_video_tiles, 0.75, exempt=True)
    idx, counts = _block_indices_from_scores(
        mx.array(scores),
        geo.num_prefix_tiles,
        geo.num_video_tiles,
        0.75,
        True,
    )
    idx_np = np.array(idx)
    counts_np = np.array(counts)
    video_mask = mask[:, geo.num_prefix_tiles:]
    for head in range(2):
        for qtile in range(geo.num_video_tiles):
            selected = set(idx_np[head, qtile, :int(counts_np[head, qtile])].tolist())
            expected = set(np.flatnonzero(video_mask[head, qtile]).tolist())
            assert selected == expected


def test_dense_first_and_dense_layer_scheduling() -> None:
    config = MiniMaxH3VSAConfig(enabled=True, sparsity=0.9, dense_first_n_steps=2, dense_layers=(3, 7))
    assert config.layer_sparsity(0, 0) == 0.0
    assert config.layer_sparsity(0, 1) == 0.0
    assert config.layer_sparsity(0, 2) == pytest.approx(0.9)
    assert config.layer_sparsity(3, 2) == 0.0
    assert config.layer_sparsity(4, 2) == pytest.approx(0.9)
    assert parse_dense_layers("0, 3, 7") == (0, 3, 7)
    assert parse_dense_layers("") == ()


def _tiny_qkv(seq: int, heads: int = 2, dim: int = 8, seed: int = 0):
    mx.random.seed(seed)
    q = mx.random.normal((seq, heads, dim))
    k = mx.random.normal((seq, heads, dim))
    v = mx.random.normal((seq, heads, dim))
    return q, k, v


def test_sparse_attention_parity_on_deterministic_small_tensors() -> None:
    spec = _TINY
    geo = build_h3_tile_geometry(spec["prefix_segments"], _dit_shape(spec), 256)
    q, k, v = _tiny_qkv(geo.total_seq_length, seed=3)
    out = h3_vsa_attention(q, k, v, geo, sparsity=0.5, exempt=True, impl="reference")
    assert out.shape == q.shape
    dense = h3_vsa_attention(q, k, v, geo, sparsity=0.0, exempt=True, impl="reference")
    prefix = geo.prefix_length
    assert mx.allclose(out[:prefix], dense[:prefix], atol=2e-4, rtol=2e-4).item()
    assert not mx.allclose(out[prefix:], dense[prefix:], atol=1e-5, rtol=1e-5).item()


def test_zero_gate_dense_mask_equivalence() -> None:
    spec = _TINY
    geo = build_h3_tile_geometry(spec["prefix_segments"], _dit_shape(spec), 256)
    q, k, v = _tiny_qkv(geo.total_seq_length, seed=4)
    zero_gate = mx.zeros_like(q)
    sparse = h3_vsa_attention(q, k, v, geo, sparsity=0.0, gate_compress=zero_gate, impl="reference")
    dense = h3_vsa_attention(q, k, v, geo, sparsity=0.0, gate_compress=None, impl="reference")
    assert mx.allclose(sparse, dense, atol=2e-4, rtol=2e-4).item()


def test_nonzero_gate_compression_changes_output() -> None:
    spec = _TINY
    geo = build_h3_tile_geometry(spec["prefix_segments"], _dit_shape(spec), 256)
    q, k, v = _tiny_qkv(geo.total_seq_length, seed=5)
    mx.random.seed(6)
    gate = mx.random.normal(q.shape) * 0.1
    with_gate = h3_vsa_attention(q, k, v, geo, sparsity=0.5, gate_compress=gate, impl="reference")
    without = h3_vsa_attention(q, k, v, geo, sparsity=0.5, gate_compress=None, impl="reference")
    assert not mx.allclose(with_gate, without, atol=1e-5, rtol=1e-5).item()


def test_int6_converter_round_trip_fifty_gate_matrices() -> None:
    spec = MLXQuantizationSpec.from_name("int6")
    from fastvideo.mlx_runtime.fastwan import quantization_support_error

    if quantization_support_error(spec) is not None:
        pytest.skip(f"INT6 is not supported by this MLX build: {quantization_support_error(spec)}")
    keys = expected_vsa_gate_keys(50)
    assert len(keys) == 50
    assert all(_is_quantizable(key, include_vsa=True) for key in keys)
    assert not any(_is_ignored_dense_key(key, include_vsa=True) for key in keys)
    assert all(_is_ignored_dense_key(key, include_vsa=False) for key in keys)
    mx.random.seed(7)
    snrs = []
    for key in keys:
        weight = mx.random.normal((64, 256)).astype(mx.float32)
        quantized = quantize_matrix(weight, spec)
        if hasattr(mx, "dequantize"):
            restored = mx.dequantize(
                quantized.weight,
                quantized.scales,
                quantized.biases,
                group_size=spec.group_size,
                bits=spec.bits,
                mode=spec.mode,
            )
        else:
            eye = mx.eye(weight.shape[1], dtype=weight.dtype)
            restored = mx.quantized_matmul(
                eye,
                quantized.weight,
                quantized.scales,
                quantized.biases,
                transpose=True,
                group_size=spec.group_size,
                bits=spec.bits,
                mode=spec.mode,
            ).T
        noise = mx.mean((weight - restored.astype(weight.dtype))**2)
        signal = mx.mean(weight**2)
        snr = float((10.0 * mx.log10(signal / mx.maximum(noise, 1e-12))).item())
        snrs.append(snr)
        assert key.endswith(VSA_GATE_KEY_SUFFIX)
    assert len(snrs) == 50
    assert min(snrs) > 15.0


def test_dense_legacy_checkpoint_compatible(tmp_path: Path) -> None:
    from fastvideo.mlx_runtime.minimax_h3 import MLXMiniMaxH3DiT

    config = {
        "hidden_size": 4,
        "num_attention_heads": 1,
        "attention_head_dim": 4,
        "ffn_dim": 4,
        "in_channels": 4,
        "audio_in_channels": 4,
        "patch_size": [1, 2, 2],
        "text_dim": 4,
        "freq_dim": 4,
        "time_embed_dim": 4,
        "rope_freq_dim": 1,
        "rope_theta": 10000.0,
        "norm_eps": 1e-5,
        "qk_norm_eps": 1e-5,
        "final_norm_eps": 1e-5,
    }
    dit = MLXMiniMaxH3DiT(
        {"proj_in.weight": mx.zeros((4, 8))},
        [{"attn.to_q.weight": mx.zeros((4, 4))}],
        [{"attn.to_q.weight": mx.zeros((4, 4))}],
        config,
    )
    assert not dit.vsa_capable
    out_dir = tmp_path / "dense"
    save_mlx_h3_checkpoint(dit, out_dir)
    loaded = load_mlx_h3_checkpoint(out_dir)
    assert not loaded.vsa_capable
    assert not mlx_h3_checkpoint_vsa_capable(out_dir)
    loaded.configure_vsa(MiniMaxH3VSAConfig(enabled=False))
    with pytest.raises(Exception, match="dense-only|include-vsa|to_gate_compress"):
        loaded.configure_vsa(MiniMaxH3VSAConfig(enabled=True))


def test_vsa_rejected_for_dense_only_checkpoint(tmp_path: Path) -> None:
    from fastvideo.mlx_runtime.minimax_h3_pipeline import MiniMaxH3MLXPipeline

    manifest = {
        "format_version": 1,
        "config": {"patch_size": [1, 2, 2], "in_channels": 8},
        "quantized_keys": {},
    }
    ckpt = tmp_path / "dense_ckpt"
    ckpt.mkdir()
    (ckpt / "mlx_h3_dit.json").write_text(json.dumps(manifest))
    (ckpt / "mlx_h3_dit.safetensors").write_bytes(b"")
    model_root = tmp_path / "model"
    (model_root / "vae").mkdir(parents=True)
    (model_root / "audio_vae").mkdir()
    (model_root / "vae" / "dummy.safetensors").write_bytes(b"x")
    (model_root / "audio_vae" / "dummy.safetensors").write_bytes(b"x")
    pipeline = MiniMaxH3MLXPipeline(model_root=model_root, mlx_dit_checkpoint=ckpt)
    with pytest.raises(Exception, match="dense-only|include-vsa|to_gate_compress"):
        pipeline.generate(
            "prompt",
            output_path=tmp_path / "out.mp4",
            vsa=True,
        )


def test_simd_kernel_parity_with_reference_identical_tiles() -> None:
    if not simd_kernel_available():
        pytest.skip("SIMD-group VSA kernel is not available")
    geo = build_h3_tile_geometry((16, 16), (8, 8, 8), tile_size=64)
    q, k, v = _tiny_qkv(geo.total_seq_length, heads=2, dim=128, seed=11)
    q, k, v = [x.astype(mx.bfloat16) for x in (q, k, v)]
    reference = h3_vsa_attention(q, k, v, geo, sparsity=0.5, exempt=True, impl="reference")
    from fastvideo.mlx_runtime.minimax_h3_vsa import MiniMaxH3VSAStats
    stats = MiniMaxH3VSAStats()
    simd = h3_vsa_attention(
        q.astype(mx.bfloat16),
        k.astype(mx.bfloat16),
        v.astype(mx.bfloat16),
        geo,
        sparsity=0.5,
        exempt=True,
        impl="simd",
        stats=stats,
    )
    mx.eval(simd)
    assert stats.impl == "simd" and stats.dense_fallback_reason is None
    _, valid = token_tile_and_valid(geo.variable_block_sizes, geo.tile_elems)
    # Packed rows are valid tokens only after untile.
    assert mx.allclose(simd.astype(mx.float32), reference.astype(mx.float32), atol=5e-2, rtol=5e-2).item()
    assert valid.sum() == geo.total_seq_length


def test_untile_hidden_never_consumes_padded_query_rows() -> None:
    """Pad slots exist in the tiled buffer; _untile_hidden must never gather them.

    The SIMD kernel may zero padded query rows only because this gather map is
    the exclusive consumer of tiled attention output.
    """
    geo = build_h3_tile_geometry((18, 414), (37, 15, 26), tile_size=64)
    _, valid = token_tile_and_valid(geo.variable_block_sizes, geo.tile_elems)
    pad_slots = np.flatnonzero(~valid.astype(bool))
    assert pad_slots.size > 0
    gathered = set(int(i) for i in geo.untile_combined_index.tolist())
    assert gathered.isdisjoint(int(i) for i in pad_slots.tolist())
    assert geo.untile_combined_index.shape == (geo.total_seq_length, )

    buf = np.ones((geo.padded_length, 2, 4), dtype=np.float32)
    buf[~valid.astype(bool)] = 999.0
    out = _untile_hidden(mx.array(buf), geo)
    mx.eval(out)
    assert out.shape[0] == geo.total_seq_length
    assert not bool(mx.any(out == 999.0).item())
    assert bool(mx.all(out == 1.0).item())


def test_simd_unsupported_tile_falls_back_to_reference() -> None:
    assert PUBLIC_VSA_IMPLS == ("auto", "reference", "simd")
    assert "metal" not in PUBLIC_VSA_IMPLS
    assert resolve_impl("simd", head_dim=128, tile_elems=256) == "reference"
    assert resolve_impl("simd", head_dim=64, tile_elems=64) == "reference"
    assert resolve_impl("auto", head_dim=128, tile_elems=64) == "reference"
    geo = build_h3_tile_geometry((8, 0, 8), (8, 8, 8), tile_size=256)
    q, k, v = _tiny_qkv(geo.total_seq_length, heads=2, dim=128, seed=12)
    explicit = h3_vsa_attention(q, k, v, geo, sparsity=0.5, exempt=True, impl="reference")
    fallback = h3_vsa_attention(q, k, v, geo, sparsity=0.5, exempt=True, impl="simd")
    assert mx.allclose(explicit, fallback, atol=0, rtol=0).item()


def test_reference_gather_path_matches_token_mask() -> None:
    """Force the chunked gather fallback (n_tiles > 24) against the full-mask path."""
    geo = build_h3_tile_geometry((8, 0, 8), (16, 8, 16), tile_size=64)
    assert geo.num_tiles > 24
    q, k, v = _tiny_qkv(geo.total_seq_length, seed=9)
    gather = h3_vsa_attention(q, k, v, geo, sparsity=0.5, exempt=True, impl="reference")
    from fastvideo.mlx_runtime import minimax_h3_vsa as vsa_mod

    previous = vsa_mod._REFERENCE_FULL_MASK_TILE_LIMIT
    vsa_mod._REFERENCE_FULL_MASK_TILE_LIMIT = geo.num_tiles + 1
    try:
        masked = h3_vsa_attention(q, k, v, geo, sparsity=0.5, exempt=True, impl="reference")
    finally:
        vsa_mod._REFERENCE_FULL_MASK_TILE_LIMIT = previous
    assert mx.allclose(gather, masked, atol=2e-4, rtol=2e-4).item()


def test_reference_full_mask_is_bounded_by_materialized_size() -> None:
    from fastvideo.mlx_runtime import minimax_h3_vsa as vsa_mod

    small = build_h3_tile_geometry((8, 0, 8), (4, 8, 8), tile_size=64)
    assert vsa_mod._reference_full_mask_fits(small, heads=40)

    # Twenty-four 256-token tiles pass the tile-count gate, but a 40-head
    # token mask would contain more than 1.5 billion elements.
    large = build_h3_tile_geometry((256, ), (4, 8, 184), tile_size=256)
    assert large.num_tiles == 24
    assert not vsa_mod._reference_full_mask_fits(large, heads=40)


def test_prefix_segments_from_t2va_layout_drop_empty_condition() -> None:
    layout = MiniMaxH3PackedLayout(
        sequence_length=20,
        position_ids=np.zeros((20, 3)),
        token_tags=np.zeros(20, dtype=np.int64),
        video_indices=np.arange(12, 20),
        audio_indices=np.arange(6, 12),
        text_indices=np.arange(6),
        num_condition_video_rows=0,
        num_condition_audio_rows=0,
        num_video_latent_frames=2,
        latent_height=4,
        latent_width=4,
        num_audio_latents=3,
    )
    assert prefix_segments_from_layout(layout, (1, 2, 2)) == (6, 0, 6)


def test_token_tile_and_valid_covers_pad_slots() -> None:
    geo = build_h3_tile_geometry((7, 5, 3), (8, 4, 6), 256)
    tile, valid = token_tile_and_valid(geo.variable_block_sizes, geo.tile_elems)
    assert tile.shape == valid.shape == (geo.padded_length, )
    assert int(valid.sum()) == geo.total_seq_length
