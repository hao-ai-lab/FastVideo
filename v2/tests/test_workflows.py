"""Cross-model Workflow convenience builders."""
from __future__ import annotations

import numpy as np

from v2.core.request import DiffusionParams, TaskType, make_request
from v2.recipes import build_image_video_engine, build_t2i_then_i2v_workflow, register_workflows
from v2.runtime import Engine


def test_t2i_then_i2v_workflow_runs_and_declares_requirements():
    eng = build_image_video_engine()
    wf = build_t2i_then_i2v_workflow()

    assert wf.requires == ["flux-t2i", "wan-i2v"]
    out = wf.run(eng, prompt="a fox", seed=3)

    assert out.artifacts["video"].frames is not None


def test_registered_workflow_is_servable_by_id():
    eng = register_workflows(Engine(), only=["image_video.t2i_i2v"])
    out = eng.run(make_request(TaskType.T2V, "image_video.t2i_i2v", "a lighthouse",
                               diffusion=DiffusionParams(seed=3)))

    assert eng.serves("image_video.t2i_i2v")
    assert np.asarray(out.artifacts["video"].frames).shape == (3, 2, 32, 48)
