# SPDX-License-Identifier: Apache-2.0
"""``Kandinsky6DenoisingStage`` guidance guard for pi-Flow (distilled) checkpoints.

A ``PiflowScheduler`` bundle runs without classifier-free guidance, so the stage refuses any ``guidance_scale`` other
than 1.0; the error must tell the user what to change. The stage is built without ``__init__`` and every call stops at
the first guard or at the missing-timesteps check, so no model is needed.
"""
from __future__ import annotations

import types

import pytest

from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from fastvideo.models.schedulers.scheduling_piflow import PiflowScheduler
from fastvideo.pipelines.stages.kandinsky6 import Kandinsky6DenoisingStage


def _stage(scheduler) -> Kandinsky6DenoisingStage:
    stage = Kandinsky6DenoisingStage.__new__(Kandinsky6DenoisingStage)
    stage.scheduler = scheduler
    return stage


def _batch(guidance_scale: float) -> types.SimpleNamespace:
    return types.SimpleNamespace(guidance_scale=guidance_scale, timesteps=None)


@pytest.mark.parametrize("guidance_scale", [5.0, 2.0, 0.0, float("nan"), float("inf")])
def test_piflow_rejects_any_guidance_but_one_with_an_actionable_message(guidance_scale):
    stage = _stage(PiflowScheduler(nfe=16, n_grid=8, shift=5.0))

    with pytest.raises(ValueError) as excinfo:
        stage.forward(_batch(guidance_scale), None)

    message = str(excinfo.value)
    assert "guidance_scale=1.0" in message
    assert "kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers" in message
    assert "kandinsky6_ti2va_distilled" in message
    # num_inference_steps is not enforced -- neither Diffusers nor FastVideo
    # requires it to equal the scheduler's nfe (PiflowScheduler.set_timesteps
    # accepts any step count and ignores nfe); the message must not claim
    # otherwise, only that 16 is the value the checkpoint was distilled for.
    assert "num_inference_steps is not constrained" in message
    assert "distilled for" in message


@pytest.mark.parametrize("guidance_scale", [1.0, 1.0 + 1e-7])
def test_piflow_accepts_guidance_one(guidance_scale):
    stage = _stage(PiflowScheduler(nfe=16, n_grid=8, shift=5.0))

    with pytest.raises(ValueError, match="timesteps must be prepared"):
        stage.forward(_batch(guidance_scale), None)


def test_flow_matching_scheduler_keeps_classifier_free_guidance():
    stage = _stage(FlowMatchEulerDiscreteScheduler(shift=5.0))

    with pytest.raises(ValueError, match="timesteps must be prepared"):
        stage.forward(_batch(5.0), None)
