"""Inference paths: Wan2.1 T2V, LTX2.3 two-stage distilled, Wan-causal streaming (design_v3 §15)."""
from __future__ import annotations

import numpy as np

from v2.models import build_default_engine
from v2.models.ltx2 import BASE_SIGMAS, REFINE_SIGMAS, build_ltx2_card
from v2.models.wan21 import build_wan21_card
from v2.models.wan_causal import build_wan_causal_card
from v2.request import DiffusionParams, OutputSpec, TaskType, make_request


def _eng():
    return build_default_engine()


def test_wan_t2v_produces_video_and_latents():
    eng = _eng()
    req = make_request(TaskType.T2V, "wan2.1-1.3b", "a cat",
                       diffusion=DiffusionParams(num_steps=4, seed=1))
    out = eng.run(req)
    assert out.artifacts["video"].frames is not None
    assert out.artifacts["latents"].latent is not None
    assert out.metrics["denoise_steps"] == 4.0


def test_ltx2_two_stage_step_counts():
    eng = _eng()
    out = eng.run(make_request(TaskType.T2V, "ltx2.3-distilled", "a sunset",
                               diffusion=DiffusionParams(seed=2)))
    # 8-step base + 3-step refine = the distilled schedule
    assert out.metrics["base_steps"] == float(len(BASE_SIGMAS) - 1) == 8.0
    assert out.metrics["refine_steps"] == float(len(REFINE_SIGMAS) - 1) == 3.0
    assert out.artifacts["video"].frames is not None


def test_causal_chunk_rollout_frame_count():
    eng = _eng()
    out = eng.run(make_request(TaskType.T2V, "wan-causal-sf-1.3b", "a river",
                               diffusion=DiffusionParams(seed=3)))
    assert out.metrics["chunks"] == 3.0
    # 3 chunks × 2 frames/chunk concatenated over the temporal axis
    assert out.artifacts["latents"].latent.shape[1] == 6


def test_determinism_same_seed_bit_identical():
    eng = _eng()
    r = lambda: make_request(TaskType.T2V, "wan2.1-1.3b", "x", diffusion=DiffusionParams(num_steps=4, seed=5))
    a = eng.run(r()).artifacts["latents"].latent
    b = eng.run(r()).artifacts["latents"].latent
    assert np.array_equal(a, b)


def test_different_seed_differs():
    eng = _eng()
    a = eng.run(make_request(TaskType.T2V, "wan2.1-1.3b", "x",
                             diffusion=DiffusionParams(num_steps=4, seed=1))).artifacts["latents"].latent
    b = eng.run(make_request(TaskType.T2V, "wan2.1-1.3b", "x",
                             diffusion=DiffusionParams(num_steps=4, seed=2))).artifacts["latents"].latent
    assert not np.array_equal(a, b)


def test_streaming_emits_chunks_only_when_requested():
    eng = _eng()
    off = eng.run(make_request(TaskType.T2V, "wan2.1-1.3b", "x",
                               diffusion=DiffusionParams(num_steps=4, seed=1)))
    on = eng.run(make_request(TaskType.T2V, "wan2.1-1.3b", "x",
                              diffusion=DiffusionParams(num_steps=4, seed=1),
                              outputs=OutputSpec(stream={"video": True})))
    assert off.metrics.get("stream_chunks", 0) == 0
    assert on.metrics.get("stream_chunks", 0) == 4    # one preview per denoise step


def test_all_cards_recipe_runtime_bound():
    # the (recipe, runtime) pair: every card's recipe assumes a loop the card declares
    for build in (build_wan21_card, build_ltx2_card, build_wan_causal_card):
        card = build()
        assert card.recipe.assumes_loop in card.loops
        assert card.parity.interleave_required is True
