# SPDX-License-Identifier: Apache-2.0
"""Registry and preset contracts for LTX-2.5 launch inference."""

import pytest

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.fastvideo_args import WorkloadType
from fastvideo.registry import get_preset_selection, get_registered_models_with_workloads


@pytest.mark.parametrize(
    ("model_path", "preset_name", "ancestral"),
    (
        ("FastVideo/LTX-2.5-Dev-Diffusers", "ltx2_5_dev", False),
        ("FastVideo/LTX-2.5-Distilled-Diffusers", "ltx2_5_distilled_two_stage", True),
    ),
)
def test_ltx2_5_registry_precedence_and_presets(
    model_path: str,
    preset_name: str,
    ancestral: bool,
) -> None:
    assert get_preset_selection(model_path) == (preset_name, "ltx2")
    registered = {entry["id"]: entry for entry in get_registered_models_with_workloads()}
    assert set(registered[model_path]["workload_types"]) == {
        WorkloadType.T2V.value,
        WorkloadType.I2V.value,
    }

    sampling = SamplingParam.from_pretrained(model_path)
    assert sampling.ltx2_image_crf == 18.0
    assert sampling.ltx2_use_ancestral_sampler is ancestral


def test_legacy_ltx_distilled_remains_deterministic() -> None:
    sampling = SamplingParam.from_pretrained("FastVideo/LTX2-Distilled-Diffusers")

    assert sampling.ltx2_use_ancestral_sampler is False
    assert sampling.ltx2_image_crf == 33.0


def test_ltx2_5_metadata_variant_selects_preset(tmp_path) -> None:
    """Local directories with fastvideo_ltx2_variant metadata route to the correct preset."""
    import json
    from pathlib import Path

    # Create a local directory with LTX2Pipeline _class_name and distilled variant metadata
    distilled_dir = tmp_path / "ltx2_5_distilled_local"
    distilled_dir.mkdir()
    model_index = {
        "_class_name": "LTX2Pipeline",
        "fastvideo_ltx2_variant": "ltx2.5-distilled",
    }
    (distilled_dir / "model_index.json").write_text(json.dumps(model_index))

    preset, family = get_preset_selection(str(distilled_dir))
    assert preset == "ltx2_5_distilled_two_stage"
    assert family == "ltx2"

    # Create a local directory with LTX2Pipeline _class_name and dev variant metadata
    dev_dir = tmp_path / "ltx2_5_dev_local"
    dev_dir.mkdir()
    model_index = {
        "_class_name": "LTX2Pipeline",
        "fastvideo_ltx2_variant": "ltx2.5-dev",
    }
    (dev_dir / "model_index.json").write_text(json.dumps(model_index))

    preset, family = get_preset_selection(str(dev_dir))
    assert preset == "ltx2_5_dev"
    assert family == "ltx2"


def test_ltx2_5_neutral_path_with_class_name_routes_correctly(tmp_path) -> None:
    """Neutral local paths with _class_name=LTX2Pipeline and variant metadata route correctly."""
    import json

    # A path with no ltx-2.5 tokens but LTX2Pipeline class and distilled variant
    neutral_dir = tmp_path / "my_converted_checkpoint"
    neutral_dir.mkdir()
    model_index = {
        "_class_name": "LTX2Pipeline",
        "fastvideo_ltx2_variant": "ltx2.5-distilled",
    }
    (neutral_dir / "model_index.json").write_text(json.dumps(model_index))

    preset, family = get_preset_selection(str(neutral_dir))
    assert preset == "ltx2_5_distilled_two_stage"
    assert family == "ltx2"
