# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import dataclasses
import importlib
import pkgutil
from pathlib import Path

import yaml

from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.configs.sample.base import SamplingParam
from fastvideo.entrypoints.cli.generate import GenerateSubcommand
from fastvideo.entrypoints.cli.serve import ServeSubcommand
from fastvideo.entrypoints.openai import image_api, video_api
from fastvideo.entrypoints.openai.protocol import (
    ImageGenerationsRequest,
    VideoGenerationsRequest,
)
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.utils import FlexibleArgumentParser


_REPO_ROOT = Path(__file__).resolve().parents[3]
_INVENTORY_PATH = _REPO_ROOT / "docs" / "design" / "inference_schema_parity_inventory.yaml"


def _load_inventory() -> dict:
    with open(_INVENTORY_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _flatten_status_section(section: dict, valid_statuses: set[str]) -> set[str]:
    names: set[str] = set()
    for status, entries in section.items():
        assert status in valid_statuses, f"Unknown status {status!r} in parity inventory"
        if isinstance(entries, dict):
            names.update(entries)
        elif isinstance(entries, list):
            names.update(entries)
        else:
            raise TypeError(f"Unsupported inventory entry type for {status!r}: {type(entries)!r}")
    return names


def _get_extra_dataclass_fields(package_name: str, base_cls: type) -> set[str]:
    package = importlib.import_module(package_name)
    base_fields = {f.name for f in dataclasses.fields(base_cls)}
    extras: set[str] = set()
    for _, modname, _ in pkgutil.iter_modules(package.__path__):
        if modname == "__pycache__":
            continue
        module = importlib.import_module(f"{package_name}.{modname}")
        for obj in vars(module).values():
            if (
                isinstance(obj, type)
                and dataclasses.is_dataclass(obj)
                and issubclass(obj, base_cls)
                and obj is not base_cls
            ):
                extras.update(f.name for f in dataclasses.fields(obj) if f.name not in base_fields)
    return extras


def _get_cli_dests(cmd_cls: type) -> set[str]:
    parser = FlexibleArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser")
    command = cmd_cls()
    subparser = command.subparser_init(subparsers)
    return {
        action.dest
        for action in subparser._actions
        if action.option_strings and action.dest != "help"
    }


def test_inventory_file_exists() -> None:
    assert _INVENTORY_PATH.exists()


def test_inventory_statuses_are_known() -> None:
    inventory = _load_inventory()
    valid_statuses = set(inventory["status_definitions"])
    for section in inventory["surfaces"].values():
        unknown = set(section) - valid_statuses
        assert not unknown, f"Unknown statuses in surface inventory: {sorted(unknown)}"


def test_fastvideo_args_fields_are_classified() -> None:
    inventory = _load_inventory()
    expected = {f.name for f in dataclasses.fields(FastVideoArgs)}
    actual = _flatten_status_section(
        inventory["surfaces"]["fastvideo_args"],
        set(inventory["status_definitions"]),
    )
    assert actual == expected


def test_pipeline_config_base_fields_are_classified() -> None:
    inventory = _load_inventory()
    expected = {f.name for f in dataclasses.fields(PipelineConfig)}
    actual = _flatten_status_section(
        inventory["surfaces"]["pipeline_config_base"],
        set(inventory["status_definitions"]),
    )
    assert actual == expected


def test_pipeline_config_extension_fields_are_classified() -> None:
    inventory = _load_inventory()
    expected = _get_extra_dataclass_fields("fastvideo.configs.pipelines", PipelineConfig)
    actual = _flatten_status_section(
        inventory["surfaces"]["pipeline_config_extensions"],
        set(inventory["status_definitions"]),
    )
    assert actual == expected


def test_sampling_param_base_fields_are_classified() -> None:
    inventory = _load_inventory()
    expected = {f.name for f in dataclasses.fields(SamplingParam)}
    actual = _flatten_status_section(
        inventory["surfaces"]["sampling_param_base"],
        set(inventory["status_definitions"]),
    )
    assert actual == expected


def test_sampling_param_extension_fields_are_classified() -> None:
    inventory = _load_inventory()
    expected = _get_extra_dataclass_fields("fastvideo.configs.sample", SamplingParam)
    actual = _flatten_status_section(
        inventory["surfaces"]["sampling_param_extensions"],
        set(inventory["status_definitions"]),
    )
    assert actual == expected


def test_openai_request_fields_are_classified() -> None:
    inventory = _load_inventory()
    valid_statuses = set(inventory["status_definitions"])

    image_expected = set(ImageGenerationsRequest.model_fields)
    image_actual = _flatten_status_section(
        inventory["surfaces"]["openai_image_request"],
        valid_statuses,
    )
    assert image_actual == image_expected

    video_expected = set(VideoGenerationsRequest.model_fields)
    video_actual = _flatten_status_section(
        inventory["surfaces"]["openai_video_request"],
        valid_statuses,
    )
    assert video_actual == video_expected


def test_cli_dest_inventory_matches_live_parsers() -> None:
    inventory = _load_inventory()

    generate_expected = set(inventory["cli"]["generate"]["expected_dests"])
    assert generate_expected == _get_cli_dests(GenerateSubcommand)

    serve_expected = set(inventory["cli"]["serve"]["expected_dests"])
    assert serve_expected == _get_cli_dests(ServeSubcommand)


def test_review_gap_fields_are_explicitly_inventory_tracked() -> None:
    inventory = _load_inventory()

    sampling_extensions = inventory["surfaces"]["sampling_param_extensions"]
    assert "guidance_scale_2" in sampling_extensions["moved"]

    image_request = inventory["surfaces"]["openai_image_request"]
    video_request = inventory["surfaces"]["openai_video_request"]
    assert "true_cfg_scale" in image_request["moved"]
    assert "guidance_scale_2" in video_request["moved"]
    assert "true_cfg_scale" in video_request["moved"]


def test_openai_size_mapping_preserves_width_height_ordering(
    monkeypatch,
    tmp_path,
) -> None:
    inventory = _load_inventory()

    monkeypatch.setattr(image_api, "get_output_dir", lambda: str(tmp_path))
    image_kwargs = image_api._build_generation_kwargs(
        request_id="img-test",
        prompt="test",
        size="640x360",
    )
    assert image_kwargs["width"] == 640
    assert image_kwargs["height"] == 360

    image_size = inventory["surfaces"]["openai_image_request"]["moved"]["size"]
    video_size = inventory["surfaces"]["openai_video_request"]["moved"]["size"]
    assert image_size["target"] == "request.sampling.width,height"
    assert video_size["target"] == "request.sampling.width,height"


def test_openai_seconds_mapping_preserves_duration_semantics(
    monkeypatch,
    tmp_path,
) -> None:
    inventory = _load_inventory()

    monkeypatch.setattr(video_api, "get_output_dir", lambda: str(tmp_path))
    request = VideoGenerationsRequest(prompt="test", seconds=4, fps=24)
    kwargs = video_api._build_generation_kwargs("vid-test", request)
    assert kwargs["fps"] == 24
    assert kwargs["num_frames"] == 96

    seconds_entry = inventory["surfaces"]["openai_video_request"][
        "compatibility_only"
    ]["seconds"]
    assert seconds_entry["target"] == "request.sampling.num_frames"
    assert "fps * seconds" in seconds_entry["note"]
