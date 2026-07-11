# SPDX-License-Identifier: Apache-2.0
"""Regression tests for None-sentinel SamplingConfig semantics.

A directly-constructed ``GenerationRequest`` must inherit the model
preset (``SamplingParam.from_pretrained``) for every sampling field the
caller did not set. Before the None-sentinel change, direct construction
marked every schema default as explicit and stomped the preset —
e.g. FastWan's 3 distilled steps silently became 50.
"""
from __future__ import annotations

import pytest

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.api.schema import GenerationRequest, SamplingConfig
from fastvideo.api.translation import (
    normalize_generation_request,
    request_to_sampling_param,
)


def _resolve(request: GenerationRequest) -> SamplingParam:
    # Mirror the production path: generate() normalizes (binding explicit
    # paths) before translating to a SamplingParam.
    return request_to_sampling_param(normalize_generation_request(request), model_path=MODEL)

# Distilled preset whose values differ from every old schema default
# (3 steps vs 50, gs 3.0 vs 1.0, 448x832 vs 720x1280, 61 frames vs 125,
# fps 16 vs 24) — if inheritance breaks, every assertion below fails.
MODEL = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"


@pytest.fixture()
def preset() -> SamplingParam:
    return SamplingParam.from_pretrained(MODEL)


class TestDirectConstructionInheritsPreset:

    def test_bare_request_preserves_preset(self, preset: SamplingParam) -> None:
        resolved = _resolve(GenerationRequest(prompt="a fox"))
        assert resolved.num_inference_steps == preset.num_inference_steps == 3
        assert resolved.guidance_scale == preset.guidance_scale == 3.0
        assert (resolved.height, resolved.width) == (preset.height, preset.width) == (448, 832)
        assert resolved.num_frames == preset.num_frames
        assert resolved.fps == preset.fps == 16
        assert resolved.negative_prompt == preset.negative_prompt
        assert resolved.negative_prompt  # preset prompt is non-empty

    def test_partial_sampling_overrides_only_set_fields(self, preset: SamplingParam) -> None:
        resolved = _resolve(GenerationRequest(prompt="a fox", sampling=SamplingConfig(num_frames=81)))
        assert resolved.num_frames == 81
        assert resolved.num_inference_steps == preset.num_inference_steps
        assert resolved.guidance_scale == preset.guidance_scale
        assert (resolved.height, resolved.width) == (preset.height, preset.width)

    def test_explicit_value_equal_to_old_schema_default_wins(self) -> None:
        # gs=1.0 was the old schema default; it must still be honored
        # when the caller sets it deliberately against a gs=3.0 preset.
        resolved = _resolve(GenerationRequest(prompt="a fox", sampling=SamplingConfig(guidance_scale=1.0)))
        assert resolved.guidance_scale == 1.0

    def test_negative_prompt_none_inherits_empty_string_clears(self, preset: SamplingParam) -> None:
        inherited = _resolve(GenerationRequest(prompt="x"))
        assert inherited.negative_prompt == preset.negative_prompt
        cleared = _resolve(GenerationRequest(prompt="x", negative_prompt=""))
        assert cleared.negative_prompt == ""

    def test_parsed_dict_request_also_inherits(self) -> None:
        from fastvideo.api.parser import parse_config
        resolved = _resolve(GenerationRequest(prompt="a fox"))
        parsed_resolved = _resolve(parse_config(GenerationRequest, {"prompt": "a fox"}))
        assert parsed_resolved.num_inference_steps == resolved.num_inference_steps
        assert parsed_resolved.guidance_scale == resolved.guidance_scale

    def test_explicitly_set_unsupported_field_still_raises(self) -> None:
        # true_cfg_scale exists on SamplingConfig but not on Wan's
        # SamplingParam; setting it explicitly must fail loudly (the old
        # schema-default tolerance is gone).
        with pytest.raises(ValueError, match="true_cfg_scale"):
            _resolve(GenerationRequest(prompt="x", sampling=SamplingConfig(true_cfg_scale=2.0)))


class TestParsedNullsInherit:
    """YAML/JSON `null` must behave exactly like an unset field — it is
    parsed into None but never bound as an explicit path."""

    def test_parsed_sampling_null_inherits_preset(self, preset: SamplingParam) -> None:
        from fastvideo.api.parser import parse_config
        parsed = parse_config(GenerationRequest, {"prompt": "x", "sampling": {"num_frames": None, "height": None}})
        resolved = _resolve(parsed)
        assert resolved.num_frames == preset.num_frames
        assert resolved.height == preset.height

    def test_parsed_negative_prompt_null_inherits_preset(self, preset: SamplingParam) -> None:
        from fastvideo.api.parser import parse_config
        parsed = parse_config(GenerationRequest, {"prompt": "x", "negative_prompt": None})
        assert _resolve(parsed).negative_prompt == preset.negative_prompt

    def test_direct_none_valued_extension_is_dropped_not_resurrected(self) -> None:
        # A None-valued extension entry is pruned from the explicit paths;
        # it must not resurface via the live-attribute read in
        # explicit_request_updates and raise as an unsupported field.
        request = GenerationRequest(prompt="x", extensions={"bogus_key": None})
        resolved = _resolve(request)
        assert resolved.num_inference_steps == 3  # preset intact, no ValueError
