# SPDX-License-Identifier: Apache-2.0
"""Request-local text refinement must preserve denoising and refresh per request."""
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from fastvideo.mlx_runtime import minimax_h3 as h3
from fastvideo.mlx_runtime import minimax_h3_pipeline as pipeline_module


def _toy_dit():
    rng = np.random.default_rng(11)

    def array(shape):
        return mx.array(rng.normal(0, 0.05, shape).astype(np.float32))

    hidden, ffn = 16, 24
    config = dict(hidden_size=hidden, num_attention_heads=1, attention_head_dim=hidden,
                  ffn_dim=ffn, in_channels=4, audio_in_channels=4, patch_size=[1, 1, 1],
                  text_dim=8, freq_dim=hidden, time_embed_dim=hidden, rope_freq_dim=2,
                  rope_theta=10000., norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5)
    weights = {}
    for name, source, target in (("context_embedder", 8, hidden), ("proj_in", 4, hidden),
                                 ("audio_proj_in", 4, hidden), ("proj_out", hidden, 4),
                                 ("audio_proj_out", hidden, 4),
                                 ("time_embedder.linear_1", hidden, hidden),
                                 ("time_embedder.linear_2", hidden, hidden),
                                 ("norm_out.linear", hidden, 2 * hidden)):
        weights[name + ".weight"] = array((target, source))
        weights[name + ".bias"] = array((target,))
    weights["token_refiner.final_norm.weight"] = mx.ones((hidden,))
    weights["norm_out.norm.weight"] = mx.ones((hidden,))

    def block():
        result = {name: mx.ones((hidden,)) for name in
                  ("norm1.weight", "norm2.weight", "attn.norm_q.weight", "attn.norm_k.weight")}
        for name in ("attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight", "attn.to_out.0.weight"):
            result[name] = array((hidden, hidden))
        result["ff.net.0.proj.weight"] = array((2 * ffn, hidden))
        result["ff.net.2.weight"] = array((hidden, ffn))
        result["adaln_proj.linear.weight"] = array((18 * hidden, hidden))
        result["adaln_proj.linear.bias"] = array((18 * hidden,))
        return result

    return h3.MLXMiniMaxH3DiT(weights, [block()], [block()], config)


def test_precomputed_refiner_is_exact_and_runs_once_for_eight_steps(monkeypatch):
    dit = _toy_dit()
    layout = h3.build_packed_layout(3, 1, 2, 2, 2, patch_size=dit.patch_size)
    timesteps, inverse = h3.build_row_timesteps(layout, .7, .4)
    dit.precompute_adaln(timesteps)
    text = mx.array(np.arange(24, dtype=np.float32).reshape(3, 8) / 24)
    video = mx.ones((len(layout.video_indices), 4))
    audio = mx.ones((len(layout.audio_indices), 4))
    calls = []
    original = dit.refine_text

    def counted(rows):
        calls.append(rows)
        return original(rows)

    monkeypatch.setattr(dit, "refine_text", counted)
    kwargs = dict(layout=layout, step_timesteps=timesteps, row_timestep_inverse=inverse)
    expected = [dit.forward_with_cache(video * (step + 1), audio, text, **kwargs) for step in range(8)]
    mx.eval(expected)
    assert len(calls) == 8
    calls.clear()
    embeds = dit.refine_text(text)
    mx.eval(embeds)
    actual = [dit.forward_with_cache(video * (step + 1), audio, text,
                                    precomputed_text_embeds=embeds, **kwargs) for step in range(8)]
    mx.eval(actual)
    assert len(calls) == 1
    for reference, cached in zip(expected, actual, strict=True):
        for left, right in zip(reference, cached, strict=True):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    with pytest.raises(ValueError, match="shape"):
        dit.forward_with_cache(video, audio, text, precomputed_text_embeds=embeds[:1], **kwargs)
    with pytest.raises(ValueError, match="floating-point"):
        dit.forward_with_cache(video, audio, text, precomputed_text_embeds=embeds.astype(mx.int32), **kwargs)


def test_pipeline_refines_again_for_next_request(monkeypatch, tmp_path):
    dit = _toy_dit()
    video_schedule = h3.MiniMaxH3SchedulerState.create(10, 8)
    audio_schedule = h3.MiniMaxH3SchedulerState.create(3, 8)
    union = h3.adaln_timestep_union(video_schedule, audio_schedule)
    schedule = SimpleNamespace(video=video_schedule, audio=audio_schedule, adaln_timesteps=union,
                               num_steps=8, pdd_plan=None)
    monkeypatch.setattr(pipeline_module, "resolve_h3_denoise_schedule", lambda *a: schedule)
    monkeypatch.setattr(pipeline_module, "_cleanup_mlx", lambda: None)
    pipeline = object.__new__(pipeline_module.MiniMaxH3MLXPipeline)
    pipeline.resident = False
    pipeline.model_root = tmp_path
    pipeline.resolve_geometry = lambda *a, **k: dict(num_frames=124, latent_frame_count=1,
                                                   latent_height=2, latent_width=2)
    calls = []
    original = dit.refine_text

    def counted(rows):
        calls.append(np.asarray(rows))
        return original(rows)

    monkeypatch.setattr(dit, "refine_text", counted)
    first_text = np.arange(24, dtype=np.float32).reshape(3, 8) / 24
    second_text = first_text[:, ::-1].copy()
    kwargs = dict(height=32, width=32, num_frames=124, seed=0, num_steps=8, dit=dit)
    first = pipeline.denoise(first_text, np.ones(3, dtype=np.int64), **kwargs)
    second = pipeline.denoise(second_text, np.ones(3, dtype=np.int64), **kwargs)
    repeated = pipeline.denoise(first_text, np.ones(3, dtype=np.int64), **kwargs)
    assert len(calls) == 3
    np.testing.assert_array_equal(calls[0], first_text)
    np.testing.assert_array_equal(calls[1], second_text)
    assert not np.array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[0], repeated[0])
    np.testing.assert_array_equal(first[1], repeated[1])
