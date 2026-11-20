# SPDX-License-Identifier: Apache-2.0
"""PAN2 falls back to the release's default negative prompt, rendered for the clip; no weights or CUDA needed."""

import json
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.configs.pipelines.pan2 import PAN2PipelineConfig
from fastvideo.pipelines.basic.pan2.negative_prompt import (PAN2_DEFAULT_NEGATIVE_PROMPT,
                                                            render_pan2_default_negative_prompt)
from fastvideo.pipelines.basic.pan2.stages import PAN2InputValidationStage, PAN2TextEncodingStage
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.text_encoding import TextEncodingStage


def _time_ranges(node):
    if isinstance(node, dict):
        return [v for k, v in node.items() if k == "time_range_sec"] + sum(map(_time_ranges, node.values()), [])
    if isinstance(node, list):
        return sum(map(_time_ranges, node), [])
    return []


def test_the_default_spans_the_clip_and_is_otherwise_the_release_template():
    rendered = json.loads(render_pan2_default_negative_prompt(481, 24))
    assert _time_ranges(rendered) == ["0 to 20"] * 3
    assert _time_ranges(PAN2_DEFAULT_NEGATIVE_PROMPT) == ["0 to 10"] * 3
    assert render_pan2_default_negative_prompt(313, 24) == json.dumps(
        json.loads(render_pan2_default_negative_prompt(313, 24)), ensure_ascii=False, separators=(",", ":"))
    assert _time_ranges(json.loads(render_pan2_default_negative_prompt(313, 24))) == ["0 to 13"] * 3


def test_single_space_preservation_is_a_pan2_policy_not_a_global_change():
    assert PAN2PipelineConfig.preserve_single_space_negative_prompt is True
    assert PipelineConfig.preserve_single_space_negative_prompt is False
    assert "preserve_single_space_negative_prompt" not in {field.name for field in fields(PAN2PipelineConfig)}


@pytest.fixture
def encode(monkeypatch):
    seen = []
    monkeypatch.setattr(TextEncodingStage, "forward", lambda self, batch, args: seen.append(batch.negative_prompt))
    stage = PAN2TextEncodingStage.__new__(PAN2TextEncodingStage)

    def run(negative_prompt, cfg=True, num_frames=121, fps=24):
        batch = SimpleNamespace(negative_prompt=negative_prompt, do_classifier_free_guidance=cfg,
                                num_frames=num_frames, fps=fps)
        stage.forward(batch, None)
        return seen[-1]

    return run


@pytest.mark.parametrize("negative_prompt", ["", None])
def test_an_empty_negative_prompt_becomes_the_default(encode, negative_prompt):
    assert encode(negative_prompt) == render_pan2_default_negative_prompt(121, 24)
    assert encode(negative_prompt, num_frames=[241], fps=[24]) == render_pan2_default_negative_prompt(241, 24)
    assert encode(negative_prompt, fps=None) == render_pan2_default_negative_prompt(121, 24)


def test_a_given_negative_prompt_or_a_space_is_kept(encode):
    assert encode("blurry") == "blurry"
    assert encode(" ") == " "


def test_without_guidance_the_negative_prompt_is_left_alone(encode):
    assert encode("", cfg=False) == ""


@pytest.mark.parametrize("negative_prompt", [None, "", " ", "blurry"])
def test_validation_then_text_encoding_resolves_the_negative_prompt(monkeypatch, negative_prompt):
    batch = ForwardBatch(
        data_type="video",
        prompt="a red fox",
        negative_prompt=negative_prompt,
        height=32,
        width=32,
        num_frames=5,
        fps=24,
        seed=0,
        num_inference_steps=2,
        guidance_scale=2.5,
    )
    args = SimpleNamespace(pipeline_config=SimpleNamespace(
        text_encoder_configs=[SimpleNamespace()],
        ti2v_task=False,
        is_causal=False,
        vae_config=SimpleNamespace(arch_config=SimpleNamespace(spatial_compression_ratio=8,
                                                               temporal_compression_ratio=4)),
        dit_config=SimpleNamespace(arch_config=SimpleNamespace(patch_size=(1, 2, 2))),
    ))
    stage = PAN2TextEncodingStage([object()], [object()])
    prompts = []

    def encode_text(prompt, *args, **kwargs):
        prompts.append(prompt)
        return [torch.ones(1, 2, 3)], [torch.ones(1, 2, dtype=torch.long)]

    monkeypatch.setattr(stage, "encode_text", encode_text)
    assert PAN2InputValidationStage().forward(batch, args) is batch
    assert stage.forward(batch, args) is batch

    expected = negative_prompt or render_pan2_default_negative_prompt(5, 24)
    assert batch.negative_prompt == expected
    assert prompts == ["a red fox", expected]
    assert len(batch.prompt_embeds) == len(batch.negative_prompt_embeds) == 1
