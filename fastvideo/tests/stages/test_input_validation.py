# SPDX-License-Identifier: Apache-2.0
"""Weight-free regression tests for negative conditioning validation."""

from types import SimpleNamespace

import pytest
import torch

from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.input_validation import InputValidationStage
from fastvideo.pipelines.stages.text_encoding import TextEncodingStage


@pytest.fixture(autouse=True)
def disable_gpu_stage_timing(monkeypatch):
    monkeypatch.setattr(
        "fastvideo.pipelines.stages.base.envs.FASTVIDEO_STAGE_LOGGING",
        SimpleNamespace(get=lambda: False),
    )


def _batch(**overrides):
    kwargs = dict(
        data_type="video",
        prompt="a red fox",
        negative_prompt=None,
        height=16,
        width=16,
        num_frames=1,
        seed=0,
        num_inference_steps=2,
        guidance_scale=5.0,
    )
    kwargs.update(overrides)
    return ForwardBatch(**kwargs)


def _args(verify):
    return SimpleNamespace(
        enable_stage_verification=verify,
        pipeline_config=SimpleNamespace(ti2v_task=False, is_causal=False),
    )


@pytest.mark.parametrize("verify", [False, True])
@pytest.mark.parametrize("negative_prompt_embeds", [None, []])
def test_cfg_rejects_missing_negative_conditioning(verify, negative_prompt_embeds):
    batch = _batch(negative_prompt_embeds=negative_prompt_embeds)
    assert batch.do_classifier_free_guidance
    assert batch.negative_prompt_embeds == []
    with pytest.raises(ValueError, match="For classifier-free guidance, either `negative_prompt`"):
        InputValidationStage()(batch, _args(verify))


@pytest.mark.parametrize("verify", [False, True])
def test_cfg_accepts_empty_negative_prompt(verify):
    batch = _batch(negative_prompt="")
    assert InputValidationStage()(batch, _args(verify)) is batch
    assert batch.negative_prompt == ""


@pytest.mark.parametrize("verify", [False, True])
def test_cfg_accepts_precomputed_negative_embeddings(verify):
    embeddings = [torch.ones(1, 2, 3)]
    batch = _batch(negative_prompt_embeds=embeddings)
    assert InputValidationStage()(batch, _args(verify)) is batch
    assert batch.negative_prompt_embeds is embeddings
    assert torch.equal(batch.negative_prompt_embeds[0], torch.ones(1, 2, 3))


@pytest.mark.parametrize("verify", [False, True])
@pytest.mark.parametrize("negative_prompt_embeds", [None, []])
def test_without_cfg_does_not_require_negative_conditioning(verify, negative_prompt_embeds):
    batch = _batch(guidance_scale=1.0, negative_prompt_embeds=negative_prompt_embeds)
    assert not batch.do_classifier_free_guidance
    assert InputValidationStage()(batch, _args(verify)) is batch


def test_empty_negative_prompt_reaches_text_encoding(monkeypatch):
    batch = _batch(negative_prompt="")
    args = _args(False)
    args.pipeline_config.text_encoder_configs = [SimpleNamespace()]
    encoder = TextEncodingStage([object()], [object()])
    prompts = []

    def encode_text(prompt, *args, **kwargs):
        prompts.append(prompt)
        return [torch.ones(1, 2, 3)], [torch.ones(1, 2, dtype=torch.long)]

    monkeypatch.setattr(encoder, "encode_text", encode_text)
    InputValidationStage()(batch, args)
    assert encoder(batch, args) is batch
    assert prompts == ["a red fox", ""]
    assert len(batch.prompt_embeds) == len(batch.negative_prompt_embeds) == 1
