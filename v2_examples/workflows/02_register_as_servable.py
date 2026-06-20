#!/usr/bin/env python3
"""02 — A workflow is a first-class servable (naming & registration).

A `Workflow` is named and registered with the same discipline as a card: a namespaced `workflow_id`
(`image_video.t2i_i2v`) in the SAME servable namespace as a `model_id`, with declared+validated
`requires`. Once registered it is addressable exactly like a model — `engine.serves(id)` reports it and
`engine.run(request)` routes to it (designv4 §9.6). This shows the per-engine registry, the fail-fast
validation, the declarative catalog (`register_workflows`), and the `WorkflowRegistry`.

Run:  python3 v2_examples/workflows/02_register_as_servable.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from v2._vendor.models import build_image_video_engine, build_t2i_then_i2v_workflow, register_workflows
from v2.core.program import WorkflowRegistry
from v2.core.request import DiffusionParams, TaskType, make_request
from v2.runtime import Engine


def main() -> None:
    # build_image_video_engine registers the cards AND the workflow as a servable
    eng = build_image_video_engine()
    print("serves('image_video.t2i_i2v'):", eng.serves("image_video.t2i_i2v"), "(a workflow, addressable like a model)")
    print("serves('flux-t2i'):", eng.serves("flux-t2i"), "(its required card)")

    # address the workflow BY ID — same call shape as any model
    out = eng.run(make_request(TaskType.T2V, "image_video.t2i_i2v", "a lighthouse",
                               diffusion=DiffusionParams(seed=3)))
    print("run-by-id video:", np.asarray(out.artifacts["video"].frames).shape)

    # declared + validated dependencies: registering on a bare engine fails fast
    try:
        Engine().register_workflow(build_t2i_then_i2v_workflow())
        print("FAIL: expected validation error")
    except ValueError as e:
        print("fail-fast validation:", "requires unregistered" in str(e))

    # the declarative catalog + the standalone registry
    fresh = register_workflows(Engine(), only=["image_video.t2i_i2v"])      # brings up cards + workflow
    print("register_workflows → serves:", fresh.serves("image_video.t2i_i2v"))
    reg = WorkflowRegistry()
    reg.register("image_video.t2i_i2v", build_t2i_then_i2v_workflow)
    print("WorkflowRegistry names:", reg.names(), "| requires:", reg.build("image_video.t2i_i2v").requires)


if __name__ == "__main__":
    main()
