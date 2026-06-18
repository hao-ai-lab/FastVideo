"""Cross-model T2I→I2V workflow — composition across model *instances* (design_v3 §13).

LTX-2 already proves same-card multi-stage diffusion (base→refine). This proves the harder case the
single-instance ``Program`` runner cannot express: chaining two *different* models (a T2I model and an
I2V model — the realistic FLUX→Wan pipeline) via a ``Workflow``, threading the image artifact between
them, with the I2V stage genuinely conditioned on the generated image. The per-model interleave-parity
guarantee is preserved because cross-model is a Workflow boundary, not a program loop step.
"""
from __future__ import annotations

import numpy as np
import pytest

from v2.recipes import build_image_video_engine, build_t2i_then_i2v_workflow, register_workflows
from v2.parity import assert_interleave_parity
from v2.program import Workflow, WorkflowRegistry, WorkflowStage
from v2.request import DiffusionParams, TaskType, make_request
from v2.request.modalpart import ImagePart, TextPart
from v2.runtime import Engine


def _i2v(eng, image, *, prompt="a scene", seed=9):
    req = make_request(TaskType.I2V, "wan-i2v", prompt,
                       inputs=(TextPart(prompt), ImagePart(pixels=image)),
                       diffusion=DiffusionParams(num_steps=4, num_frames=81, seed=seed))
    return eng.run(req).artifacts["video"].frames


def _t2i(eng, prompt, seed):
    req = make_request(TaskType.T2I, "flux-t2i", prompt,
                       diffusion=DiffusionParams(num_steps=4, num_frames=1, seed=seed))
    return eng.run(req).artifacts["image"].tensor


# --- the workflow ----------------------------------------------------------------- #

def test_t2i_then_i2v_workflow_produces_video():
    eng = build_image_video_engine()
    out = build_t2i_then_i2v_workflow().run(eng, prompt="a red fox in snow", seed=3)
    frames = np.asarray(out.artifacts["video"].frames)
    assert frames.ndim == 4 and frames.shape[1] > 1            # a multi-frame video, not a still


def test_workflow_crosses_two_distinct_instances():
    eng = build_image_video_engine()
    a = eng._registry["flux-t2i"][0]
    b = eng._registry["wan-i2v"][0]
    assert a is not b
    assert a.card.model_id != b.card.model_id                 # two separate models / weight sets
    assert a.component("transformer") is not b.component("transformer")


def test_i2v_stage_consumes_the_image():
    """The hand-off is real: swapping the conditioning image changes the video; the same image is
    deterministic."""
    eng = build_image_video_engine()
    img1 = _t2i(eng, "alpha", seed=1)
    img2 = _t2i(eng, "totally different scene", seed=2)
    assert not np.array_equal(np.asarray(img1), np.asarray(img2))
    v1 = _i2v(eng, img1)
    v2 = _i2v(eng, img2)
    assert not np.array_equal(np.asarray(v1), np.asarray(v2))  # image changes the video
    assert np.array_equal(np.asarray(_i2v(eng, img1)), np.asarray(v1))  # same image ⇒ identical


def test_workflow_propagates_prompt_to_both_stages():
    eng = build_image_video_engine()
    a = build_t2i_then_i2v_workflow().run(eng, prompt="alpha", seed=1).artifacts["video"].frames
    b = build_t2i_then_i2v_workflow().run(eng, prompt="beta gamma delta", seed=1).artifacts["video"].frames
    assert not np.array_equal(np.asarray(a), np.asarray(b))


# --- the two cards are independently usable + each preserves its parity gate ------- #

def test_each_stage_runs_standalone():
    eng = build_image_video_engine()
    img = _t2i(eng, "a lighthouse", seed=5)
    assert np.asarray(img).shape[1] == 1                       # T2I ⇒ single frame (an image)
    vid = _i2v(eng, img)
    assert np.asarray(vid).shape[1] > 1                        # I2V ⇒ video


def test_each_model_still_passes_interleave_parity():
    eng = build_image_video_engine()
    t2i = [make_request(TaskType.T2I, "flux-t2i", p,
                        diffusion=DiffusionParams(num_steps=4, num_frames=1, seed=s))
           for p, s in [("a", 1), ("b", 2), ("a", 1)]]
    assert not assert_interleave_parity(eng, t2i)


def test_workflow_rejects_model_mismatch():
    """A stage that builds a request for the wrong model is a programming error, caught early."""
    eng = build_image_video_engine()
    bad = Workflow("bad", [WorkflowStage(
        "flux-t2i",
        lambda s: make_request(TaskType.I2V, "wan-i2v", "x"))])   # declares flux-t2i, builds wan-i2v
    with pytest.raises(ValueError):
        bad.run(eng, prompt="x")


# --- naming + registration: a workflow is a first-class servable ------------------ #

def test_workflow_is_addressable_by_its_id_like_a_model():
    """The workflow_id is a servable name in the SAME namespace as model ids: serves() reports it and
    engine.run routes to it — so the OpenAI server / fleet address it exactly like a card."""
    eng = build_image_video_engine()
    assert eng.serves("image_video.t2i_i2v")                     # registered as a servable
    assert eng.serves("flux-t2i") and eng.serves("wan-i2v")      # its required cards too
    out = eng.run(make_request(TaskType.T2V, "image_video.t2i_i2v", "a fox",
                               diffusion=DiffusionParams(seed=3)))
    assert np.asarray(out.artifacts["video"].frames).shape[1] > 1


def test_workflow_declares_and_validates_required_cards():
    """``requires`` is declared (the stage model_ids); registration fails fast if a card is absent."""
    wf = build_t2i_then_i2v_workflow()
    assert wf.requires == ["flux-t2i", "wan-i2v"]
    with pytest.raises(ValueError, match="requires unregistered"):
        Engine().register_workflow(wf)                           # no cards registered → fail fast


def test_workflow_id_must_not_collide_with_a_model_id():
    eng = build_image_video_engine()
    clash = Workflow("flux-t2i", [WorkflowStage("flux-t2i", lambda s: None)])  # id == a card id
    with pytest.raises(ValueError, match="collides"):
        eng.register_workflow(clash)


def test_workflow_registry_catalog():
    reg = WorkflowRegistry()
    reg.register("image_video.t2i_i2v", build_t2i_then_i2v_workflow)
    assert "image_video.t2i_i2v" in reg and reg.names() == ["image_video.t2i_i2v"]
    assert reg.build("image_video.t2i_i2v").requires == ["flux-t2i", "wan-i2v"]
    with pytest.raises(ValueError, match="already registered"):
        reg.register("image_video.t2i_i2v", build_t2i_then_i2v_workflow)
    with pytest.raises(KeyError):
        reg.build("nonexistent.workflow")


def test_register_workflows_helper_brings_up_cards_and_workflow():
    """The declarative catalog: one call registers the workflow and the cards it needs."""
    eng = register_workflows(Engine(), only=["image_video.t2i_i2v"])
    assert eng.serves("image_video.t2i_i2v")
    out = eng.run(make_request(TaskType.T2V, "image_video.t2i_i2v", "a city",
                               diffusion=DiffusionParams(seed=1)))
    assert out.artifacts["video"].frames is not None
