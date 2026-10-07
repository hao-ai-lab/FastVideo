# SPDX-License-Identifier: Apache-2.0
"""Regression tests for ``Kandinsky6CFGResolutionStage``, which recomputes
``batch.do_classifier_free_guidance`` under Kandinsky6's own CFG contract, mirroring the diffusers
reference's (``pipeline_kandinsky6_ti2va.py``) ``do_classifier_free_guidance`` property: the uncond pass
runs whenever ``guidance_scale > 1.0``, same as the generic rule (``ForwardBatch.__post_init__``); any
guidance_scale <= 1.0 (including below 1.0) runs cond-only with no negative-prompt encoding. A PiFlow
(distilled) scheduler never runs CFG regardless of the requested guidance_scale.

Pure logic test -- no GPU, no model weights.
"""
from __future__ import annotations

import types

import pytest

from fastvideo.pipelines.stages.kandinsky6 import Kandinsky6CFGResolutionStage


def _stage(scheduler) -> Kandinsky6CFGResolutionStage:
    stage = Kandinsky6CFGResolutionStage.__new__(Kandinsky6CFGResolutionStage)
    stage.scheduler = scheduler
    return stage


def _flow_scheduler() -> types.SimpleNamespace:
    return types.SimpleNamespace(is_piflow=False)


def _piflow_scheduler() -> types.SimpleNamespace:
    return types.SimpleNamespace(is_piflow=True)


@pytest.mark.parametrize("guidance_scale,expected", [
    (5.0, True),
    (2.0, True),
    (1.0 + 1e-5, True),
    (1.0, False),
    (1.0 - 1e-9, False),
    (0.5, False),
    (0.0, False),
])
def test_cfg_trigger_uses_guidance_scale_greater_than_1(guidance_scale, expected):
    stage = _stage(_flow_scheduler())
    batch = types.SimpleNamespace(guidance_scale=guidance_scale, do_classifier_free_guidance=None)

    out = stage.forward(batch, fastvideo_args=None)

    assert out.do_classifier_free_guidance is expected


@pytest.mark.parametrize("guidance_scale", [1.0, 5.0, 0.0, float("nan")])
def test_piflow_scheduler_never_triggers_cfg(guidance_scale):
    stage = _stage(_piflow_scheduler())
    batch = types.SimpleNamespace(guidance_scale=guidance_scale, do_classifier_free_guidance=None)

    out = stage.forward(batch, fastvideo_args=None)

    assert out.do_classifier_free_guidance is False


def test_nan_guidance_on_a_flow_scheduler_does_not_trigger_cfg():
    stage = _stage(_flow_scheduler())
    batch = types.SimpleNamespace(guidance_scale=float("nan"), do_classifier_free_guidance=None)

    out = stage.forward(batch, fastvideo_args=None)

    assert out.do_classifier_free_guidance is False
