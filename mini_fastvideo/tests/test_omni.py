"""Phase-2 omni: one resident MoT instance runs AR + diffusion loops on shared weights.

This is the §16 claim — true omni/MoT serving: one resident model, many loop types, scheduled at
step granularity, in one request — made native (not a DAG doubling weights, not a monolith bypassing
the abstraction). Cosmos3 = reason → joint denoise; BAGEL/lance = generate_text → generate_image.
"""
from __future__ import annotations

import numpy as np

from mini_fastvideo._enums import WorkUnitKind
from mini_fastvideo.models import build_omni_engine
from mini_fastvideo.models.cosmos3 import build_cosmos3_card
from mini_fastvideo.parity import assert_interleave_parity
from mini_fastvideo.request import DiffusionParams, SamplingParams, TaskType, make_request


def _eng():
    return build_omni_engine()


def _omni_req(model_id, task, prompt, seed, max_tokens=6, steps=4):
    return make_request(task, model_id, prompt,
                        sampling=SamplingParams(max_tokens=max_tokens, seed=seed),
                        diffusion=DiffusionParams(num_steps=steps, seed=seed))


def test_cosmos3_reasoner_then_joint_denoise_one_request():
    eng = _eng()
    out = eng.run(_omni_req("cosmos3-vfm", TaskType.T2V, "a phoenix", 1))
    assert out.artifacts["video"].frames is not None       # diffusion pathway produced video
    assert out.artifacts["text"].text.startswith("tok:")   # und pathway produced reasoned tokens
    assert out.metrics["tokens"] > 0 and out.metrics["denoise_steps"] > 0


def test_bagel_generate_text_then_image_one_request():
    eng = _eng()
    out = eng.run(_omni_req("bagel-mot", TaskType.T2I, "a teapot", 2))
    assert out.artifacts["text"].text.startswith("tok:")   # AR generate_text
    assert out.artifacts["image"].tensor is not None       # diffusion generate_image
    assert out.metrics["tokens"] > 0 and out.metrics["denoise_steps"] > 0


def test_mot_one_resident_instance_shared_by_both_loops():
    eng = _eng()
    inst = eng._registry["cosmos3-vfm"][0]
    t = inst.component("transformer")
    assert inst.component("transformer") is t              # exactly one resident copy
    # both loops declare they bind the same component (no weight duplication, no DAG split)
    assert "transformer" in inst.card.loops["ar_decode"].shared_weight_components
    assert "transformer" in inst.card.loops["diffusion_denoise"].shared_weight_components
    # and the one module exposes BOTH pathways
    assert hasattr(t, "ar_forward") and callable(t)


def test_omni_scheduler_sees_both_work_unit_kinds():
    eng = _eng()
    eng.run(_omni_req("cosmos3-vfm", TaskType.T2V, "a comet", 3))
    by_kind = eng.admission.metrics.by_kind
    # the scheduler priced BOTH AR tokens and denoise steps — runtime-visible, not one opaque stage
    assert by_kind[WorkUnitKind.AR_TOKEN.value] > 0
    assert by_kind[WorkUnitKind.DIFFUSION_STEP.value] > 0


def test_omni_interleave_parity_holds_across_loop_types():
    eng = _eng()
    reqs = [_omni_req("cosmos3-vfm", TaskType.T2V, "alpha", 11),
            _omni_req("cosmos3-vfm", TaskType.T2V, "beta", 22)]
    assert not assert_interleave_parity(eng, reqs)         # ar+denoise interleave is still bit-identical


def test_cosmos3_sound_vae_is_optional_declared_not_loaded():
    # the lazy-component fix (P8): sound_vae is declared optional_for non-t2vs, never an env-var hack
    card = build_cosmos3_card()
    assert "t2v" in card.components["sound_vae"].optional_for
    assert "t2vs" in card.components["sound_vae"].required_for


def test_omni_determinism():
    eng = _eng()
    a = eng.run(_omni_req("bagel-mot", TaskType.T2I, "x", 5)).artifacts["image"].tensor
    b = eng.run(_omni_req("bagel-mot", TaskType.T2I, "x", 5)).artifacts["image"].tensor
    assert np.array_equal(a, b)
