"""The batch-of-N interleave parity gate (design_v3 §9.3) — the bet loop-inversion lives/dies on.

Two+ concurrent requests, interleaved at step granularity, MUST be bit-identical to the same
requests run serially. A batch-of-1 gate is structurally blind to cross-request state smearing;
this gate is not.
"""
from __future__ import annotations

from v2.models import build_default_engine
from v2.parity import assert_interleave_parity
from v2.request import DiffusionParams, TaskType, make_request


def _req(mid, prompt, seed, steps=4):
    return make_request(TaskType.T2V, mid, prompt,
                        diffusion=DiffusionParams(num_steps=steps, guidance_scale=5.0, seed=seed))


def test_interleave_parity_all_three_models():
    eng = build_default_engine()
    for mid in eng._registry:
        reqs = [_req(mid, "alpha", 11), _req(mid, "beta", 22), _req(mid, "gamma", 33)]
        divs = assert_interleave_parity(eng, reqs)
        assert not divs, f"{mid} interleave gate FAILED: {divs}"


def test_interleave_parity_shared_prompt_and_cache():
    # shared prompt exercises the cross-request feature cache; must still be bit-identical
    eng = build_default_engine()
    reqs = [_req("wan2.1-1.3b", "shared", 7), _req("wan2.1-1.3b", "shared", 7),
            _req("wan2.1-1.3b", "shared", 9)]
    assert not assert_interleave_parity(eng, reqs)


def test_interleave_parity_mixed_step_counts():
    # a 2-step and an 8-step request interleave; fairness, not corruption
    eng = build_default_engine()
    reqs = [_req("wan2.1-1.3b", "short", 1, steps=2), _req("wan2.1-1.3b", "long", 2, steps=8)]
    assert not assert_interleave_parity(eng, reqs)
