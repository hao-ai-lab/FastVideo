# SPDX-License-Identifier: Apache-2.0
import yaml

from fastvideo.api import (
    apply_overrides,
    load_run_config,
    parse_cli_overrides,
)


def test_parse_cli_overrides_casts_supported_scalar_and_collection_types() -> None:
    parsed = parse_cli_overrides([
        "--generator.engine.num_gpus",
        "4",
        "--request.runtime.enable_teacache=true",
        "--request.sampling.guidance_scale",
        "1.5",
        "--request.prompt",
        "[\"a\", \"b\"]",
        "--request.extensions",
        "{\"ltx2\": {\"initial_latent_path\": \"/tmp/init.pt\"}}",
        "--request.output.output_video_name",
        "clip",
        "--request.state",
        "null",
    ])

    assert parsed == {
        "generator.engine.num_gpus": 4,
        "request.runtime.enable_teacache": True,
        "request.sampling.guidance_scale": 1.5,
        "request.prompt": ["a", "b"],
        "request.extensions": {"ltx2": {"initial_latent_path": "/tmp/init.pt"}},
        "request.output.output_video_name": "clip",
        "request.state": None,
    }


def test_parse_cli_overrides_normalizes_dashed_dotted_keys() -> None:
    parsed = parse_cli_overrides([
        "--generator.engine.num-gpus",
        "2",
        "--request.output.output-path",
        "outputs/custom.mp4",
    ])

    assert parsed == {
        "generator.engine.num_gpus": 2,
        "request.output.output_path": "outputs/custom.mp4",
    }


def test_apply_overrides_merges_nested_dicts_without_mutating_source() -> None:
    original = {
        "generator": {
            "model_path": "/models/base",
            "engine": {"num_gpus": 1},
        },
        "request": {},
    }

    updated = apply_overrides(
        original,
        {
            "generator.engine.num_gpus": 8,
            "request.extensions.ltx2.initial_latent_path": "/tmp/init.pt",
        },
    )

    assert original["generator"]["engine"]["num_gpus"] == 1
    assert updated["generator"]["engine"]["num_gpus"] == 8
    assert updated["request"]["extensions"]["ltx2"]["initial_latent_path"] == "/tmp/init.pt"


def test_load_run_config_applies_dotted_overrides_before_validation(tmp_path) -> None:
    raw = {
        "generator": {"model_path": "/models/base"},
        "request": {"prompt": "baseline"},
    }
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    loaded = load_run_config(path, overrides=[
        "--generator.pipeline.workload_type",
        "t2v",
        "--request.sampling.num_frames",
        "81",
        "--request.extensions.ltx2.initial_latent_path",
        "/tmp/latent.pt",
    ])

    assert loaded.generator.pipeline.workload_type == "t2v"
    assert loaded.request.sampling.num_frames == 81
    assert loaded.request.extensions["ltx2"]["initial_latent_path"] == "/tmp/latent.pt"
