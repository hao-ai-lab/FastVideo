# SPDX-License-Identifier: Apache-2.0
from fastvideo.api import (
    GenerationRequest,
    GeneratorConfig,
    RunConfig,
    SamplingConfig,
    ServeConfig,
    config_to_dict,
)


def test_run_config_roundtrip_preserves_nested_defaults() -> None:
    config = RunConfig(
        generator=GeneratorConfig(model_path="hf://model"),
        request=GenerationRequest(
            prompt="hello",
            sampling=SamplingConfig(num_frames=48, width=832, height=480),
        ),
    )

    dumped = config_to_dict(config)

    assert dumped["generator"]["model_path"] == "hf://model"
    assert dumped["generator"]["engine"]["execution_backend"] == "mp"
    assert dumped["request"]["sampling"]["num_frames"] == 48
    assert dumped["request"]["sampling"]["guidance_scale_2"] is None
    assert dumped["request"]["output"]["save_video"] is True


def test_serve_config_includes_server_and_default_request_defaults() -> None:
    config = ServeConfig(generator=GeneratorConfig(model_path="/models/ltx2"))

    dumped = config_to_dict(config)

    assert dumped["server"] == {
        "host": "0.0.0.0",
        "port": 8000,
        "output_dir": "outputs/",
    }
    assert dumped["default_request"]["sampling"]["fps"] == 24
    assert dumped["default_request"]["runtime"]["enable_teacache"] is False
