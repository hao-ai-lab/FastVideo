#!/usr/bin/env python3
"""01 — Cross-model T2I → I2V workflow (FLUX→Wan-style).

Two *separate* models chained by a `Workflow`: `flux-t2i` (text → image) → `wan-i2v` (text + image →
video). The workflow runs a full `engine.run` per stage and threads the stage-1 image artifact into the
stage-2 request as an `ImagePart`; the I2V program folds the image into its conditioning. Each model is
a distinct instance with its own weights — composition across model *instances*, not just loops within
one (designv4 §9.6).

Run:  python3 v2_examples/workflows/01_t2i_then_i2v.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from v2.recipes import build_image_video_engine, build_t2i_then_i2v_workflow


def main() -> None:
    eng = build_image_video_engine()                 # registers flux-t2i + wan-i2v (+ the workflow)
    wf = build_t2i_then_i2v_workflow()
    print("workflow:", wf.workflow_id, "| requires cards:", wf.requires)

    out = wf.run(eng, prompt="a red fox in snow", seed=3)
    video = np.asarray(out.artifacts["video"].frames)
    print(f"  T2I → I2V video: {video.shape}")

    # the I2V stage genuinely consumes the generated image: a different prompt → different image →
    # different video (the hand-off is real, not cosmetic)
    a = np.asarray(build_t2i_then_i2v_workflow().run(eng, prompt="alpha", seed=1).artifacts["video"].frames)
    b = np.asarray(build_t2i_then_i2v_workflow().run(eng, prompt="beta gamma", seed=1).artifacts["video"].frames)
    print(f"  different prompt → different video: {not np.array_equal(a, b)}")
    print("  (stage 2 is conditioned on stage 1's image — composition across two model instances)")


if __name__ == "__main__":
    main()
