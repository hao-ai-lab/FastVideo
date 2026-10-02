# SPDX-License-Identifier: Apache-2.0
"""Request resolution records the source of each setting and agrees with the SamplingParam built for the request."""
from dataclasses import fields

import pytest

from fastvideo.api.compat import normalize_generation_request, request_to_sampling_param
from fastvideo.api.request_resolution import resolve_request
from fastvideo.api.schema import OutputConfig, RequestRuntimeConfig, SamplingConfig
from fastvideo.tests.api.config_snapshot import isolated_environment

WAN_T2V = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
LTX2 = "FastVideo/LTX2-Distilled-Diffusers"


def test_request_values_and_preset_defaults_are_told_apart():
    with isolated_environment():
        request = normalize_generation_request({"prompt": "a cat", "sampling": {"seed": 7}})
        resolved = resolve_request(request, model_path=WAN_T2V)

    seed = resolved.provenance("sampling.seed")
    assert (seed.value, seed.source, seed.explicit) == (7, "input", True)
    frames = resolved.provenance("sampling.num_frames")
    assert (frames.source, frames.explicit) == ("fill_sampling_defaults[preset wan_t2v_1_3b]", False)


@pytest.mark.parametrize(("model_path", "raw"), [
    (WAN_T2V, {"prompt": "a cat"}),
    (WAN_T2V, {"prompt": "a cat", "sampling": {"num_inference_steps": 12, "guidance_scale": 4.0, "height": 720}}),
    (LTX2, {"prompt": "a cat", "output": {"save_video": False}}),
])
def test_resolved_settings_match_the_sampling_param(model_path, raw):
    with isolated_environment():
        request = normalize_generation_request(raw)
        resolved = resolve_request(request, model_path=model_path)
        sampling_param = request_to_sampling_param(request, model_path=model_path)

    for section, config_type in (("sampling", SamplingConfig), ("runtime", RequestRuntimeConfig),
                                 ("output", OutputConfig)):
        for config_field in fields(config_type):
            if hasattr(sampling_param, config_field.name):
                resolved_value = resolved.to_dict()[section][config_field.name]
                assert resolved_value == getattr(sampling_param, config_field.name), f"{section}.{config_field.name}"
