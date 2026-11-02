# SPDX-License-Identifier: Apache-2.0
"""Pika H3 Flex request constraints: the six-step ladder and the configured tier.

CPU-only; no weights are loaded.
"""

from __future__ import annotations

import pytest

from fastvideo.api.compat import legacy_generate_call_to_request, normalize_generation_request
from fastvideo.api.parser import parse_config
from fastvideo.api.pika_h3_flex import PikaH3FlexSamplingParam
from fastvideo.api.schema import GenerationRequest, SamplingConfig
from fastvideo.configs.pipelines.pika_h3_flex import PikaH3FlexPipelineConfig
from fastvideo.pipelines.basic.pika_h3_flex.buckets import resolve_canvas


def _request(sampling: dict) -> GenerationRequest:
    return parse_config(GenerationRequest, {"prompt": "a paper boat", "sampling": sampling})


def _constrain(tier: str, request: GenerationRequest) -> PikaH3FlexSamplingParam:
    # The sampling-param defaults (768x1344) are what made an omitted size look explicit.
    return PikaH3FlexPipelineConfig(pika_resolution_tier=tier).apply_request_constraints(
        request, PikaH3FlexSamplingParam())


@pytest.mark.parametrize(("tier", "canvas"), [("p480", (480, 832)), ("p768", (768, 1344))])
def test_omitted_size_takes_the_configured_tier(tier, canvas):
    sampling_param = _constrain(tier, _request({}))

    assert (sampling_param.height, sampling_param.width) == canvas
    # Input preparation then sees an exact bucket of the configured tier.
    assert resolve_canvas(sampling_param.height, sampling_param.width)[2] == tier


def test_generate_video_call_without_size_takes_the_configured_tier():
    request = legacy_generate_call_to_request("a paper boat", None, legacy_kwargs={"num_frames": 124})

    sampling_param = _constrain("p480", request)

    assert (sampling_param.height, sampling_param.width) == (480, 832)


def test_python_request_counts_its_schema_default_size_as_set():
    """A GenerationRequest built in Python carries SamplingConfig's 720x1280 as a set size."""
    request = normalize_generation_request(
        GenerationRequest(prompt="a paper boat", sampling=SamplingConfig(num_inference_steps=6)))

    sampling_param = PikaH3FlexPipelineConfig(pika_resolution_tier="p480").apply_request_constraints(
        request, PikaH3FlexSamplingParam(height=720, width=1280))

    # Explicit, so it is left for input preparation to snap (to the p768 16:9 bucket).
    assert (sampling_param.height, sampling_param.width) == (720, 1280)
    assert resolve_canvas(720, 1280)[:3] == (768, 1344, "p768")


@pytest.mark.parametrize(("tier", "size"), [("p480", (768, 1344)), ("p768", (480, 832)), ("p480", (1024, 768))])
def test_explicit_size_wins_over_the_configured_tier(tier, size):
    height, width = size
    sampling_param = PikaH3FlexPipelineConfig(pika_resolution_tier=tier).apply_request_constraints(
        _request({
            "height": height,
            "width": width
        }), PikaH3FlexSamplingParam(height=height, width=width))

    assert (sampling_param.height, sampling_param.width) == size


@pytest.mark.parametrize("sampling", [{"height": 480}, {"width": 832}])
def test_one_sided_size_is_rejected(sampling):
    with pytest.raises(ValueError, match="must be passed together"):
        _constrain("p768", _request(sampling))


def test_step_ladder_is_still_enforced():
    sampling_param = _constrain("p768", _request({}))
    assert sampling_param.num_inference_steps == 6
    with pytest.raises(ValueError, match="runs exactly 6 transformer forwards"):
        _constrain("p768", _request({"num_inference_steps": 8}))
