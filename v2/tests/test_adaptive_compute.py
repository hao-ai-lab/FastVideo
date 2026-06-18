"""Content-adaptive control flow — the loop owns it (design_v3 §2.2, §9.12).

Because the model owns control flow (the driven loop), content-adaptive compute is natural: the
`CacheDiTDenoiseLoop`'s `next()` reuses the cached velocity when consecutive predictions barely change
(cache-dit / TeaCache — skip the DiT forward), and early-exits when the latent converges — a variable,
content-dependent step count. These assert: skips happen and stay close to the full run; early-exit
reduces steps; and the interleave parity gate still holds across requests with *different* step counts.
"""
from __future__ import annotations

import numpy as np

from v2.cache import CacheManager
from v2.card import load_card
from v2.recipes import build_wan_t2v_program
from v2.recipes.adaptive import build_adaptive_card
from v2.parity import assert_interleave_parity
from v2.request import DiffusionParams, TaskType, make_request
from v2.runtime import Engine

STEPS = 12


def _engine(*, cache_threshold=0.0, exit_threshold=0.0):
    card = build_adaptive_card(cache_threshold=cache_threshold, exit_threshold=exit_threshold)
    inst = load_card(card, cache_manager=CacheManager.from_card(card))
    eng = Engine()
    eng.register(card.model_id, inst, build_wan_t2v_program())
    return eng


def _run(eng, prompt="a calm sea", seed=3):
    return eng.run(make_request(TaskType.T2V, "wan-adaptive", prompt,
                                diffusion=DiffusionParams(num_steps=STEPS, seed=seed)))


def test_threshold_zero_is_plain_denoise():
    out = _run(_engine(cache_threshold=0.0))
    assert out.metrics["denoise_steps"] == float(STEPS)
    assert out.metrics["skipped_steps"] == 0.0           # subclass with threshold 0 ⇒ identical to base


def test_cache_dit_skips_forwards_and_stays_close():
    full = _run(_engine(cache_threshold=0.0))
    cached = _run(_engine(cache_threshold=0.15))
    assert cached.metrics["skipped_steps"] > 0           # the loop chose to skip DiT forwards
    v_full = np.asarray(full.artifacts["video"].frames)
    v_cached = np.asarray(cached.artifacts["video"].frames)
    rel = float(np.linalg.norm(v_cached - v_full) / (np.linalg.norm(v_full) + 1e-8))
    assert rel < 0.05                                    # skipping is a close approximation, not exact


def test_early_exit_reduces_steps_in_the_tail():
    eng = _engine(exit_threshold=0.05)
    out = _run(eng)
    assert out.metrics["early_exited"] == 1.0
    assert out.metrics["denoise_steps"] < float(STEPS)   # stopped before the full schedule
    assert out.metrics["denoise_steps"] >= STEPS // 2    # ...but only in the low-noise tail, not at step 0


def test_interleave_parity_holds_with_variable_step_counts():
    """Different prompts skip/exit at different steps ⇒ ragged loops; serial == interleaved still."""
    eng = _engine(cache_threshold=0.15)
    reqs = [_make("alpha", 1), _make("beta gamma", 2), _make("alpha", 1)]
    assert not assert_interleave_parity(eng, reqs)


def _make(prompt, seed):
    return make_request(TaskType.T2V, "wan-adaptive", prompt,
                        diffusion=DiffusionParams(num_steps=STEPS, seed=seed))
