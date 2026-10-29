# SPDX-License-Identifier: Apache-2.0
"""PAN2 falls back to the release's default negative prompt, rendered for the clip; no weights or CUDA needed."""

import json
from types import SimpleNamespace

import pytest

from fastvideo.pipelines.basic.pan2.negative_prompt import (PAN2_DEFAULT_NEGATIVE_PROMPT,
                                                            render_pan2_default_negative_prompt)
from fastvideo.pipelines.basic.pan2.stages import PAN2TextEncodingStage
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
