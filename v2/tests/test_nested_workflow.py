"""Nested workflows — recursive cross-model composition.

A workflow id is a servable, so a `WorkflowStage` can invoke a *workflow* exactly as it invokes a
model. These tests build a workflow whose first stage IS the T2I→I2V workflow (then extends its video),
register it on an engine that already serves the inner workflow, and assert: it runs end-to-end;
`requires` lists the inner workflow; validation recurses; and a self-referencing workflow is rejected
by the cycle guard (no infinite recursion).
"""
from __future__ import annotations

import numpy as np
import pytest

from v2.recipes import build_image_video_engine, build_t2i_i2v_extend_workflow
from v2.program import Workflow, WorkflowStage
from v2.request import TaskType, make_request
from v2.runtime import Engine


def test_nested_workflow_runs_end_to_end():
    eng = build_image_video_engine()                 # serves flux-t2i, wan-i2v, AND image_video.t2i_i2v
    outer = build_t2i_i2v_extend_workflow()
    eng.register_workflow(outer)
    out = outer.run(eng, prompt="a red fox in snow", seed=2)
    frames = np.asarray(out.artifacts["video"].frames)
    assert frames.ndim == 4 and frames.shape[1] > 1   # the extended (I2V) video


def test_nested_workflow_requires_the_inner_workflow():
    outer = build_t2i_i2v_extend_workflow()
    assert outer.requires == ["image_video.t2i_i2v", "wan-i2v"]   # a workflow id among its deps
    # registering on an engine that does NOT serve the inner workflow fails fast
    with pytest.raises(ValueError, match="requires unregistered"):
        Engine().register_workflow(outer)


def test_nested_workflow_addressable_by_id():
    eng = build_image_video_engine()
    eng.register_workflow(build_t2i_i2v_extend_workflow())
    assert eng.serves("image_video.t2i_i2v_extend")
    out = eng.run(make_request(TaskType.T2V, "image_video.t2i_i2v_extend", "a lighthouse"))
    assert out.artifacts["video"].frames is not None  # engine.run routes the OUTER workflow,
    #                                                   which routes the INNER workflow as its stage 1


def _selfref():
    return Workflow("image_video.loop", [WorkflowStage(
        "image_video.loop", lambda s: make_request(TaskType.T2V, "image_video.loop", s["prompt"]))])


def test_self_reference_caught_at_registration():
    """A workflow that requires itself can't even register — `requires` includes its own (unserved) id.
    This is the primary cycle prevention (declared deps, validated)."""
    eng = build_image_video_engine()
    with pytest.raises(ValueError, match="requires unregistered"):
        eng.register_workflow(_selfref())


def test_runtime_cycle_guard_is_defense_in_depth():
    """Even if a cycle slipped past validation (here forced by inserting directly), the run-time guard
    raises rather than recursing forever."""
    eng = build_image_video_engine()
    eng._workflows["image_video.loop"] = _selfref()              # bypass validate() to simulate a cycle
    with pytest.raises(ValueError, match="cycle"):
        eng.run(make_request(TaskType.T2V, "image_video.loop", "x"))
