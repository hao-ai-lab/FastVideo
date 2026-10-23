# SPDX-License-Identifier: Apache-2.0
"""Prefix K/V reuse across denoise steps (MLX MiniMax H3 Ref2VA).

Tiny random-weight DiT: the recording step is bit-identical to the normal
forward, a reuse step whose inputs did not move equals the full forward, the
budget refuses oversized caches, and the target-only VSA attention selects the
same tiles (so produces the same rows) as the full-sequence VSA attention.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx.core", reason="MLX is required for MiniMax H3 prefix K/V tests")
pytest.importorskip("torch", reason="the tiny H3 fixture builds a torch reference model")

import mlx.core as mx  # noqa: E402

from fastvideo.mlx_runtime.minimax_h3 import (  # noqa: E402
    MINIMAX_H3_TEXT_TAG, build_ref2va_packed_layout, build_row_timesteps)
from fastvideo.mlx_runtime.minimax_h3_pipeline import MiniMaxH3MLXPipeline  # noqa: E402
from fastvideo.mlx_runtime.minimax_h3_prefix_kv import (  # noqa: E402
    MiniMaxH3PrefixKVCache, PrefixKVBudgetError, build_target_query_plan, check_prefix_kv_budget,
    estimate_prefix_kv_bytes, frozen_prefix_rows)
from fastvideo.mlx_runtime.minimax_h3_reference import MiniMaxH3PreparedReference  # noqa: E402
from fastvideo.mlx_runtime.minimax_h3_vsa import (  # noqa: E402
    MiniMaxH3VSAConfig, build_h3_segment_geometry, h3_vsa_attention)
from fastvideo.mlx_runtime.minimax_h3_vsa_target import h3_vsa_attention_target  # noqa: E402
from fastvideo.tests.mlx.tiny_h3 import (  # noqa: E402
    LATENT_HEIGHT, LATENT_WIDTH, NUM_AUDIO_LATENTS, NUM_LATENT_FRAMES, NUM_TEXT_TOKENS, TINY_ARCH, build_hf_config,
    build_tiny_h3_config, build_torch_model, mlx_dit_from_torch_model)

PATCH = tuple(TINY_ARCH["patch_size"])


def _reference(media_type: str, *, frames: int = 1, has_audio: bool = False):
    reference = MiniMaxH3PreparedReference(media_type=media_type, has_audio=has_audio)
    reference.num_latent_frames = frames
    reference.latent_height = 4
    reference.latent_width = 4
    reference.num_audio_latents = 2 if has_audio else 0
    return reference


def _layout(references=None):
    references = references or [_reference("image")]
    tags = np.full(NUM_TEXT_TOKENS, MINIMAX_H3_TEXT_TAG, dtype=np.int64)
    return build_ref2va_packed_layout(tags, references, NUM_LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH,
                                      NUM_AUDIO_LATENTS, PATCH)


@pytest.fixture(scope="module")
def tiny_dit():
    dit = mlx_dit_from_torch_model(build_torch_model(), build_hf_config(build_tiny_h3_config()))
    layout = _layout()
    dit.precompute_adaln(np.array([0.1, 0.4, 0.7, 0.999, 1.0], dtype=np.float32), drop_weights=False)
    return dit, layout


def _inputs(layout, seed=0):
    rng = np.random.default_rng(seed)
    n_video = layout.video_indices.shape[0]
    n_audio = layout.audio_indices.shape[0]
    return {
        "x_v": mx.array(rng.standard_normal((n_video, TINY_ARCH["in_channels"] * 4)).astype(np.float32)),
        "x_a": mx.array(rng.standard_normal((n_audio, TINY_ARCH["audio_in_channels"])).astype(np.float32)),
        "text": mx.array(rng.standard_normal((NUM_TEXT_TOKENS, TINY_ARCH["text_dim"])).astype(np.float32)),
    }


def _step_args(layout, video_t=0.7, audio_t=0.4):
    unique, inverse = build_row_timesteps(layout, video_t, audio_t, condition_video_timestep=0.999)
    return {"step_timesteps": unique, "row_timestep_inverse": inverse}


def _cache_for(dit, layout, storage="native"):
    return MiniMaxH3PrefixKVCache(len(dit.blocks), frozen_prefix_rows(layout), dit.num_heads, dit.head_dim, storage)


def _full(dit, layout, inputs, record=None, video_t=0.7, audio_t=0.4, step_index=0):
    kwargs = {"record_prefix_kv": record} if record is not None else {}
    video, audio = dit.forward_with_cache(inputs["x_v"], inputs["x_a"], inputs["text"], layout=layout,
                                          step_index=step_index, **_step_args(layout, video_t, audio_t), **kwargs)
    mx.eval(video, audio)
    return np.asarray(video), np.asarray(audio)


def _reuse(dit, layout, inputs, cache, video_t=0.7, audio_t=0.4, step_index=1):
    video, audio = dit.forward_target_rows(inputs["x_v"][layout.num_condition_video_rows:],
                                           inputs["x_a"][layout.num_condition_audio_rows:], layout=layout,
                                           prefix_kv=cache, step_index=step_index,
                                           **_step_args(layout, video_t, audio_t))
    mx.eval(video, audio)
    return np.asarray(video), np.asarray(audio)


def test_recording_step_matches_the_normal_forward_exactly(tiny_dit) -> None:
    dit, layout = tiny_dit
    inputs = _inputs(layout)
    plain = _full(dit, layout, inputs)
    cache = _cache_for(dit, layout)
    recorded = _full(dit, layout, inputs, record=cache)
    np.testing.assert_array_equal(recorded[0], plain[0])
    np.testing.assert_array_equal(recorded[1], plain[1])
    assert cache.complete and cache.recorded_step == 0
    assert cache.rows == layout.sequence_length - layout.audio_indices.shape[0] + layout.num_condition_audio_rows - (
        layout.video_indices.shape[0] - layout.num_condition_video_rows)
    expected = estimate_prefix_kv_bytes(len(dit.blocks), cache.rows, dit.num_heads, dit.head_dim, "native", 4)
    assert cache.nbytes == expected


def test_reuse_with_unmoved_inputs_equals_the_full_forward(tiny_dit) -> None:
    """If nothing the prefix depends on changed, frozen K/V is exactly what the full pass computes."""
    dit, layout = tiny_dit
    inputs = _inputs(layout, seed=1)
    cache = _cache_for(dit, layout)
    video_full, audio_full = _full(dit, layout, inputs, record=cache)
    video, audio = _reuse(dit, layout, inputs, cache)
    np.testing.assert_allclose(video, video_full[layout.num_condition_video_rows:], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(audio, audio_full[layout.num_condition_audio_rows:], rtol=1e-4, atol=1e-5)


def test_reuse_matches_an_independent_frozen_prefix_reference(tiny_dit) -> None:
    """Moved target rows: compare against a hand-rolled block loop that concatenates the cached prefix K/V."""
    from fastvideo.mlx_runtime.minimax_h3 import (_adaln_tables, _feed_forward, _h3_rms_norm, _modulate,
                                                  apply_h3_rotary, linear, rope_cos_sin)

    dit, layout = tiny_dit
    cache = _cache_for(dit, layout)
    _full(dit, layout, _inputs(layout, seed=2), record=cache)
    moved = _inputs(layout, seed=3)
    video, audio = _reuse(dit, layout, moved, cache, video_t=0.4, audio_t=0.1)

    frozen = cache.rows
    n_a = layout.audio_indices.shape[0] - layout.num_condition_audio_rows
    n_v = layout.video_indices.shape[0] - layout.num_condition_video_rows
    args = _step_args(layout, 0.4, 0.1)
    positions = mx.array(dit._adaln_cache.positions(args["step_timesteps"]))
    rows = positions[mx.array(args["row_timestep_inverse"][frozen:])]
    adaln = (rows * 3 + mx.array(layout.token_tags[frozen:])).astype(mx.int32)
    cos, sin = rope_cos_sin(mx.array(layout.position_ids[frozen:]), dit.rope_freq_dim, dit.rope_theta)
    h = mx.concatenate([
        linear(moved["x_a"][layout.num_condition_audio_rows:], dit.weights["audio_proj_in.weight"],
               dit.weights["audio_proj_in.bias"]),
        linear(moved["x_v"][layout.num_condition_video_rows:], dit.weights["proj_in.weight"],
               dit.weights["proj_in.bias"]),
    ])
    heads, dim = dit.num_heads, dit.head_dim
    for index, block in enumerate(dit.blocks):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = dit._adaln_cache.block_tables[index]
        normed = _modulate(_h3_rms_norm(h, block["norm1.weight"], dit.norm_eps), (1 + scale_a)[adaln], shift_a[adaln])
        q = apply_h3_rotary(
            _h3_rms_norm(linear(normed, block["attn.to_q.weight"]).reshape(-1, heads, dim),
                         block["attn.norm_q.weight"], dit.qk_norm_eps), cos, sin)
        k = apply_h3_rotary(
            _h3_rms_norm(linear(normed, block["attn.to_k.weight"]).reshape(-1, heads, dim),
                         block["attn.norm_k.weight"], dit.qk_norm_eps), cos, sin)
        v = linear(normed, block["attn.to_v.weight"]).reshape(-1, heads, dim)
        k_all = mx.concatenate([cache.load(index)[0], k])
        v_all = mx.concatenate([cache.load(index)[1], v])
        scores = mx.einsum("qhd,khd->hqk", q, k_all) * dim**-0.5
        attended = mx.einsum("hqk,khd->qhd", mx.softmax(scores, axis=-1), v_all).reshape(-1, heads * dim)
        h = h + gate_a[adaln] * linear(attended, block["attn.to_out.0.weight"])
        normed = _modulate(_h3_rms_norm(h, block["norm2.weight"], dit.norm_eps), (1 + scale_m)[adaln], shift_m[adaln])
        h = h + gate_m[adaln] * _feed_forward(block, normed)
    shift_scale = dit._adaln_cache
    normed = _h3_rms_norm(h, dit.weights["norm_out.norm.weight"], dit.final_norm_eps)
    normed = normed * (1.0 + shift_scale.norm_out_scale[rows]) + shift_scale.norm_out_shift[rows]
    ref_video = linear(normed[n_a:], dit.weights["proj_out.weight"], dit.weights.get("proj_out.bias"))
    ref_audio = linear(normed[:n_a], dit.weights["audio_proj_out.weight"], dit.weights.get("audio_proj_out.bias"))
    assert ref_video.shape[0] == n_v
    np.testing.assert_allclose(video, np.asarray(ref_video), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(audio, np.asarray(ref_audio), rtol=1e-4, atol=1e-5)


def test_storage_dtypes_stay_close_and_int8_is_smaller(tiny_dit) -> None:
    dit, layout = tiny_dit
    inputs = _inputs(layout, seed=4)
    exact = _cache_for(dit, layout, "native")
    _full(dit, layout, inputs, record=exact)
    reference = _reuse(dit, layout, inputs, exact)
    sizes = {}
    for storage, tolerance in (("bf16", 0.05), ("fp16", 0.01), ("int8", 0.1)):
        cache = _cache_for(dit, layout, storage)
        _full(dit, layout, inputs, record=cache)
        sizes[storage] = cache.nbytes
        got = _reuse(dit, layout, inputs, cache)
        scale = np.abs(reference[0]).max()
        assert np.abs(got[0] - reference[0]).max() < tolerance * scale
    assert sizes["int8"] < sizes["fp16"] == sizes["bf16"] < exact.nbytes


def test_cache_size_guards() -> None:
    estimate = estimate_prefix_kv_bytes(50, 7700, 56, 128, "bf16")
    assert 10.0 < estimate / 2**30 < 10.5  # the ~11 GB (decimal) full-canvas case
    with pytest.raises(PrefixKVBudgetError, match="prefix_kv_cache_max_gib"):
        check_prefix_kv_budget(estimate, 4.0, rows=7700)
    check_prefix_kv_budget(estimate, 12.0, rows=7700)
    assert estimate_prefix_kv_bytes(50, 7700, 56, 128, "int8") < estimate * 0.52
    with pytest.raises(ValueError):
        check_prefix_kv_budget(1, 0.0, rows=1)


def test_reuse_rejects_a_stale_or_mismatched_cache(tiny_dit) -> None:
    dit, layout = tiny_dit
    inputs = _inputs(layout)
    with pytest.raises(ValueError, match="incomplete"):
        _reuse(dit, layout, inputs, _cache_for(dit, layout))
    with pytest.raises(ValueError, match="Ref2VA"):
        from fastvideo.mlx_runtime.minimax_h3 import build_packed_layout
        frozen_prefix_rows(
            build_packed_layout(NUM_TEXT_TOKENS, NUM_LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH, NUM_AUDIO_LATENTS,
                                patch_size=PATCH))


@pytest.mark.parametrize("references", [
    [_reference("image")],
    [_reference("video", frames=2, has_audio=True), _reference("image")],
])
def test_vsa_reuse_with_unmoved_inputs_equals_the_full_forward(references) -> None:
    dit = mlx_dit_from_torch_model(build_torch_model(), build_hf_config(build_tiny_h3_config()))
    rng = np.random.default_rng(5)
    inner = dit.num_heads * dit.head_dim
    for block in dit.blocks:
        block["attn.to_gate_compress.weight"] = mx.array(
            rng.standard_normal((inner, TINY_ARCH["hidden_size"])).astype(np.float32) * 0.05)
    dit._vsa_capable = True
    dit._gate_active = [None] * len(dit.blocks)
    dit.configure_vsa(MiniMaxH3VSAConfig(enabled=True, sparsity=0.5, tile_size=64, impl="reference"))
    layout = _layout(references)
    dit.precompute_adaln(np.array([0.1, 0.4, 0.7, 0.999, 1.0], dtype=np.float32), drop_weights=False)
    inputs = _inputs(layout, seed=6)
    cache = _cache_for(dit, layout)
    video_full, audio_full = _full(dit, layout, inputs, record=cache)
    video, audio = _reuse(dit, layout, inputs, cache)
    np.testing.assert_allclose(video, video_full[layout.num_condition_video_rows:], rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(audio, audio_full[layout.num_condition_audio_rows:], rtol=1e-3, atol=1e-4)


# text, image ref, reference video (token grid), reference audio, target audio, target video (token grid)
_SEGMENTS = (40, 30, (2, 4, 8), 25, 70, (4, 6, 8))


@pytest.mark.parametrize("tile_size", [64, 128])
@pytest.mark.parametrize("sparsity,gate,keep_rate", [(0.0, False, 0.5), (0.0, True, 0.5), (0.7, False, 0.5),
                                                      (0.7, True, 0.5), (0.9, True, 0.25)])
def test_vsa_target_queries_select_the_same_tiles_as_the_full_path(tile_size, sparsity, gate, keep_rate) -> None:
    geometry = build_h3_segment_geometry(_SEGMENTS, tile_size)
    total = geometry.total_seq_length
    n_video = int(np.prod(_SEGMENTS[-1]))
    n_audio = _SEGMENTS[-2]
    frozen = total - n_video - n_audio
    heads, dim = 3, 16
    rng = np.random.default_rng(tile_size)
    q, k, v, g = (mx.array(rng.standard_normal((total, heads, dim)).astype(np.float32)) for _ in range(4))
    full = h3_vsa_attention(q, k, v, geometry, sparsity=sparsity, gate_compress=g if gate else None,
                            ref_keep_rate=keep_rate, impl="reference")
    plan = build_target_query_plan(geometry, frozen, n_audio, n_video)
    target = h3_vsa_attention_target(q[frozen:], k, v, geometry, plan, sparsity=sparsity,
                                     gate_compress=g[frozen:] if gate else None, ref_keep_rate=keep_rate,
                                     impl="reference")
    np.testing.assert_allclose(np.asarray(target), np.asarray(full[frozen:]), rtol=1e-4, atol=1e-5)
    if sparsity > 0.0:
        # Sparse routing really is exercised: dropping tiles changes the dense result.
        dense = h3_vsa_attention(q, k, v, geometry, sparsity=0.0, impl="reference")
        assert float(mx.abs(full[frozen:] - dense[frozen:]).max()) > 1e-3


def test_vsa_target_plan_validates_row_accounting() -> None:
    geometry = build_h3_segment_geometry(_SEGMENTS, 64)
    with pytest.raises(ValueError, match="do not cover"):
        build_target_query_plan(geometry, 10, 70, int(np.prod(_SEGMENTS[-1])))


class _StubDiT:
    """Minimal DiT that records which forward each denoise step used."""

    patch_size = PATCH
    patch_dim = 16
    audio_in_channels = 2
    vsa_config = None
    _adaln_cache = None
    last_vsa_stats = None
    num_heads = 8
    head_dim = 64
    blocks = [{}, {}]

    def __init__(self) -> None:
        self.calls: list[str] = []

    def precompute_adaln(self, timesteps, *, drop_weights: bool = True):
        return None

    def forward_with_cache(self, x_v, x_a, text, *, layout, step_timesteps, row_timestep_inverse, step_index=0,
                           record_prefix_kv=None):
        self.calls.append("record" if record_prefix_kv is not None else "full")
        return x_v * 0.0 + 1.0, x_a * 0.0 + 1.0

    def forward_target_rows(self, x_v, x_a, *, layout, step_timesteps, row_timestep_inverse, prefix_kv, step_index=0):
        self.calls.append("reuse")
        return x_v * 0.0 + 1.0, x_a * 0.0 + 1.0


def _denoise(tmp_path: Path, dit, num_steps=4, **kwargs):
    from fastvideo.tests.mlx.test_mlx_minimax_h3_ref2va_pipeline import (
        AUDIO_IN_CHANNELS, HEIGHT, NUM_FRAMES, NUM_TEXT_TOKENS as PIPELINE_TEXT_TOKENS, PATCH_DIM, WIDTH,
        _expected_layout, _pipeline, _references)

    pipeline = _pipeline(tmp_path)
    layout = _expected_layout()
    pipeline.denoise(
        np.zeros((PIPELINE_TEXT_TOKENS, 16), dtype=np.float32),
        np.full(PIPELINE_TEXT_TOKENS, 1, dtype=np.int64),
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        audio_num_frames=NUM_FRAMES,
        seed=0,
        num_steps=num_steps,
        dit=dit,
        references=_references(),
        condition_video_rows=np.zeros((layout.num_condition_video_rows, PATCH_DIM), dtype=np.float32),
        condition_audio_rows=np.zeros((layout.num_condition_audio_rows, AUDIO_IN_CHANNELS), dtype=np.float32),
        **kwargs,
    )
    return pipeline


def test_denoise_loop_records_then_reuses_and_refreshes(tmp_path: Path) -> None:
    dit = _StubDiT()
    pipeline = _denoise(tmp_path, dit, num_steps=4, reuse_prefix_kv=True, prefix_kv_storage="bf16")
    assert dit.calls == ["record", "reuse", "reuse", "reuse"]
    stats = pipeline.last_prefix_kv_stats
    assert stats["enabled"] and stats["recorded_steps"] == [0] and stats["reused_steps"] == [1, 2, 3]
    assert len(pipeline.last_denoise_step_times) == 4

    dit = _StubDiT()
    pipeline = _denoise(tmp_path, dit, num_steps=4, reuse_prefix_kv=True, prefix_kv_refresh_every=2)
    assert dit.calls == ["record", "reuse", "record", "reuse"]


def test_denoise_default_never_touches_prefix_kv(tmp_path: Path) -> None:
    dit = _StubDiT()
    pipeline = _denoise(tmp_path, dit, num_steps=2)
    assert dit.calls == ["full", "full"]
    assert pipeline.last_prefix_kv_stats["enabled"] is False


def test_denoise_refuses_a_cache_over_budget(tmp_path: Path) -> None:
    dit = _StubDiT()
    with pytest.raises(PrefixKVBudgetError, match="prefix_kv_cache_max_gib"):
        _denoise(tmp_path, dit, reuse_prefix_kv=True, prefix_kv_cache_max_gib=1e-9)
    assert dit.calls == []


def test_reuse_needs_references() -> None:
    pipeline = MiniMaxH3MLXPipeline.__new__(MiniMaxH3MLXPipeline)
    with pytest.raises(ValueError, match="Ref2VA"):
        pipeline.denoise(np.zeros((1, 1), dtype=np.float32), np.zeros(1, dtype=np.int64), height=64, width=64,
                         num_frames=17, seed=0, reuse_prefix_kv=True)
    with pytest.raises(ValueError, match="refresh_every"):
        pipeline.denoise(np.zeros((1, 1), dtype=np.float32), np.zeros(1, dtype=np.int64), height=64, width=64,
                         num_frames=17, seed=0, prefix_kv_refresh_every=-1)
