"""LTX-2 joint audio+video denoise — T2VS with per-modality guidance.

LTX-2 declared an `audio_vae` (`required_for={"t2vs"}`) but never used it. These tests light it up: a
single two-stage denoise carries a **synchronized audio latent** alongside the video (audio conditioned
on the video latent), applies **per-modality CFG** (`guidance_per_modality`), and decodes both via the
video VAE and the audio VAE → video + audio. The default T2V path is unchanged (gated on the request
asking for audio), so the existing LTX-2 tests are untouched.
"""
from __future__ import annotations

import numpy as np

from v2.runtime.cache import CacheManager
from v2.core.card import load_card
from v2.recipes.ltx2 import build_ltx2_av_program, build_ltx2_card
from v2.core.request import DiffusionParams, OutputSpec, TaskType, make_request
from v2.runtime import Engine

AV = frozenset({"video", "audio"})


def _engine():
    card = build_ltx2_card()
    inst = load_card(card, cache_manager=CacheManager.from_card(card))
    eng = Engine()
    eng.register(card.model_id, inst, build_ltx2_av_program())
    return eng


def _t2vs(prompt="a thunderstorm", *, seed=3, vg=3.0, ag=6.0):
    return make_request(TaskType.T2VS, "ltx2-2stage-distilled", prompt,
                        outputs=OutputSpec(modalities=AV),
                        diffusion=DiffusionParams(guidance_per_modality={"video": vg, "audio": ag}, seed=seed))


def test_ltx2_produces_video_and_audio():
    out = _engine().run(_t2vs())
    assert "video" in out.artifacts and "audio" in out.artifacts
    assert np.asarray(out.artifacts["video"].frames).ndim == 4
    samples = np.asarray(out.artifacts["audio"].samples)
    assert samples.ndim == 1 and samples.size > 0
    assert out.artifacts["audio"].sample_rate == 44100
    # the 2-stage distilled structure is preserved (base 8-step → refine 3-step)
    assert out.metrics["base_steps"] == 8.0 and out.metrics["refine_steps"] == 3.0


def test_audio_is_synced_to_the_video():
    """Joint (not independent) generation: the audio latent is conditioned on the video, so a different
    video content yields different audio; same prompt+seed is deterministic."""
    eng = _engine()
    a = np.asarray(eng.run(_t2vs("a calm lake")).artifacts["audio"].samples)
    b = np.asarray(eng.run(_t2vs("a violent explosion")).artifacts["audio"].samples)
    assert not np.array_equal(a, b)
    assert np.array_equal(a, np.asarray(eng.run(_t2vs("a calm lake")).artifacts["audio"].samples))


def test_per_modality_guidance_is_independent():
    """`guidance_per_modality` sets video and audio CFG separately: changing only the audio guidance
    changes the audio but leaves the video bit-identical, and vice-versa."""
    eng = _engine()
    base_v = np.asarray(eng.run(_t2vs(vg=3.0, ag=6.0)).artifacts["video"].frames)
    base_a = np.asarray(eng.run(_t2vs(vg=3.0, ag=6.0)).artifacts["audio"].samples)
    # bump ONLY audio guidance → audio changes, video unchanged
    only_audio = eng.run(_t2vs(vg=3.0, ag=9.0))
    assert np.array_equal(base_v, np.asarray(only_audio.artifacts["video"].frames))
    assert not np.array_equal(base_a, np.asarray(only_audio.artifacts["audio"].samples))
    # bump ONLY video guidance → video changes
    only_video = eng.run(_t2vs(vg=7.0, ag=6.0))
    assert not np.array_equal(base_v, np.asarray(only_video.artifacts["video"].frames))


def test_t2v_path_unchanged_no_audio():
    """A plain T2V request (no audio modality) produces no audio latent — the video-only path is
    untouched (this is why the existing LTX-2 tests still pass)."""
    card = build_ltx2_card()
    inst = load_card(card, cache_manager=CacheManager.from_card(card))
    eng = Engine()
    from v2.recipes.ltx2 import build_ltx2_program
    eng.register(card.model_id, inst, build_ltx2_program())
    out = eng.run(make_request(TaskType.T2V, "ltx2-2stage-distilled", "a sunset",
                               diffusion=DiffusionParams(seed=3)))
    assert "audio" not in out.artifacts
    assert np.asarray(out.artifacts["video"].frames).ndim == 4
