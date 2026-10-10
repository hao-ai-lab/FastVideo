# SPDX-License-Identifier: Apache-2.0
"""The LTX-2 refine stages follow the forward-time ``ltx2_refine_enabled``.

``LTX2Pipeline`` builds its refine stages from the load-time args (converted
LTX-2.5 checkpoints set ``fastvideo_refine_enabled`` in model_index.json), but
training validation runs ``pipeline.forward`` with args built from the training
config, where refinement is off. Every refine stage must then be a no-op;
before this, the refine denoising stage alone still ran and re-denoised the
finished stage-1 latents from the stage-2 sigma.

CPU-only: the stages are built around placeholder modules and never reach a
transformer.
"""

from types import SimpleNamespace

import pytest
import torch

from fastvideo.pipelines import ForwardBatch
from fastvideo.pipelines.basic.ltx2.ltx2_pipeline import LTX2Pipeline
from fastvideo.pipelines.basic.ltx2.stages.ltx2_denoising import LTX2DenoisingStage

_REFINE_STAGES = (
    "ltx2_refine_init_stage",
    "ltx2_upsample_stage",
    "ltx2_refine_lora_stage",
    "ltx2_refine_denoising_stage",
)


def _args(*, refine_enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(
        ltx2_refine_enabled=refine_enabled,
        ltx2_refine_num_inference_steps=3,
        ltx2_refine_add_noise=True,
        ltx2_refine_lora_path="distilled_lora/model.safetensors",
        ltx2_refine_guidance_scale=1.0,
    )


def _pipeline_with_refine_stages() -> LTX2Pipeline:
    pipeline = LTX2Pipeline.__new__(LTX2Pipeline)
    pipeline.modules = {"transformer": object()}
    pipeline._stages = []
    pipeline._stage_name_mapping = {}
    pipeline.create_pipeline_stages(_args(refine_enabled=True))
    return pipeline


def test_refine_stages_are_noops_when_refine_is_off_at_forward() -> None:
    pipeline = _pipeline_with_refine_stages()
    assert set(_REFINE_STAGES) <= set(pipeline._stage_name_mapping)

    latents = torch.randn(1, 128, 3, 2, 2)
    batch = ForwardBatch(data_type="video", latents=latents.clone(), height=64, width=64)
    for name in _REFINE_STAGES:
        stage = pipeline._stage_name_mapping[name]
        assert stage.forward(batch, _args(refine_enabled=False)) is batch, name

    assert torch.equal(batch.latents, latents)
    assert (batch.height, batch.width) == (64, 64)


def test_refine_denoising_stage_still_runs_when_refine_is_on() -> None:
    stage = _pipeline_with_refine_stages()._stage_name_mapping["ltx2_refine_denoising_stage"]
    with pytest.raises(ValueError, match="Latents must be provided"):
        stage.forward(ForwardBatch(data_type="video"), _args(refine_enabled=True))


def test_stage1_denoising_ignores_the_refine_flag() -> None:
    stage = LTX2DenoisingStage(transformer=object())
    with pytest.raises(ValueError, match="Latents must be provided"):
        stage.forward(ForwardBatch(data_type="video"), _args(refine_enabled=False))
