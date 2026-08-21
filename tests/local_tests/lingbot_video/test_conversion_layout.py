# SPDX-License-Identifier: Apache-2.0
"""CPU-only layout tests for the LingBot-Video checkpoint converter."""

import json
import os
from pathlib import Path

import pytest

from scripts.checkpoint_conversion import lingbot_video_to_diffusers as converter


def _make_source(tmp_path: Path, has_refiner: bool) -> Path:
    """Create the minimal official component tree exercised by ``convert``."""
    source = tmp_path / "source"
    for index, component_name in enumerate(converter.PASSTHROUGH_COMPONENTS):
        component = source / component_name
        component.mkdir(parents=True)
        (component / "config.json").write_text("{}\n")
        (component / "weights.safetensors").write_bytes(bytes([index]))
    for component_name in ("text_encoder", "processor"):
        component = source / component_name
        component.mkdir(parents=True)
        (component / "config.json").write_text("{}\n")
    if has_refiner:
        refiner = source / "refiner"
        refiner.mkdir()
        (refiner / "config.json").write_text("{}\n")
        (refiner / "weights.safetensors").write_bytes(b"refiner")
    return source


def _fake_convert_text_encoder(_source: Path, destination: Path) -> dict[str, str]:
    """Create a lightweight destination in place of tensor conversion."""
    destination.mkdir()
    return {"weight": "model.safetensors"}


def _fake_write_text_encoder_config(_source: Path, _destination: Path) -> dict:
    """Return the metadata consumed by the converter's completion message."""
    return {"architectures": ["LingBotVideoQwen3VLTextModel"]}


@pytest.mark.parametrize("has_refiner", [False, True])
def test_convert_emits_optional_transformer_2_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    has_refiner: bool,
) -> None:
    """Keep Dense output unchanged and map official refiners to ``transformer_2``."""
    source = _make_source(tmp_path, has_refiner)
    destination = tmp_path / "converted"
    validated_configs: list[Path] = []
    monkeypatch.setattr(converter, "_convert_text_encoder", _fake_convert_text_encoder)
    monkeypatch.setattr(converter, "_write_text_encoder_config", _fake_write_text_encoder_config)
    monkeypatch.setattr(converter, "_validate_transformer_config", validated_configs.append)

    converter.convert(source, destination)

    model_index = json.loads((destination / "model_index.json").read_text())
    expected_pipeline_class = converter.MOE_PIPELINE_CLASS if has_refiner else converter.DENSE_PIPELINE_CLASS
    assert model_index["_class_name"] == expected_pipeline_class
    assert ("transformer_2" in model_index) is has_refiner
    assert (destination / "transformer_2").is_dir() is has_refiner
    assert os.path.samefile(
        source / "transformer" / "weights.safetensors",
        destination / "transformer" / "weights.safetensors",
    )
    assert len(validated_configs) == (2 if has_refiner else 1)
    if has_refiner:
        assert model_index["transformer_2"] == ["diffusers", "LingBotVideoTransformer3DModel"]
        assert os.path.samefile(
            source / "refiner" / "weights.safetensors",
            destination / "transformer_2" / "weights.safetensors",
        )
        assert not os.path.samefile(
            destination / "transformer" / "weights.safetensors",
            destination / "transformer_2" / "weights.safetensors",
        )
