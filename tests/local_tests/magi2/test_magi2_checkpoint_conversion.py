# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-layout conversion coverage for the MAGI-2 port."""

from __future__ import annotations

import json
from pathlib import Path

from fastvideo.utils import verify_model_config_and_directory
from scripts.checkpoint_conversion.convert_magi2_to_fastvideo import (
    COMPONENT_DIRECTORY_MAPPING,
    MODEL_INDEX,
    REQUIRED_COMPONENT_FILES,
    SOURCE_REVISION,
    TURBO_VAE_SKIPPED_KEYS,
    convert_checkpoint_layout,
)


def _create_minimal_source(source: Path) -> dict[str, Path]:
    """Create identifying files for every published checkpoint component."""
    source_files: dict[str, Path] = {}
    for component, relative_paths in REQUIRED_COMPONENT_FILES.items():
        for relative_path in relative_paths:
            source_path = source / component / relative_path
            source_path.parent.mkdir(parents=True, exist_ok=True)
            if relative_path.endswith(".index.json"):
                shard_name = f"{component}-00001-of-00001.safetensors"
                source_path.write_text(
                    json.dumps({"weight_map": {"weight": shard_name}}),
                    encoding="utf-8",
                )
                shard_path = source_path.parent / shard_name
                shard_path.write_bytes(shard_name.encode())
                source_files[f"{component}/{shard_name}"] = shard_path
            else:
                source_path.write_bytes(f"{component}/{relative_path}".encode())
            source_files[f"{component}/{relative_path}"] = source_path
    return source_files


def test_magi2_converter_preserves_files_and_records_all_mappings(tmp_path: Path) -> None:
    """Require hard-linked components and complete conversion metadata."""
    source = tmp_path / "official"
    output = tmp_path / "fastvideo"
    source_files = _create_minimal_source(source)

    convert_checkpoint_layout(source, output)

    assert verify_model_config_and_directory(str(output)) == MODEL_INDEX
    for source_relative_path, source_path in source_files.items():
        source_component, component_relative_path = source_relative_path.split("/", 1)
        destination_component = COMPONENT_DIRECTORY_MAPPING[source_component]
        destination_path = output / destination_component / component_relative_path
        assert destination_path.read_bytes() == source_path.read_bytes()
        assert destination_path.stat().st_ino == source_path.stat().st_ino

    manifest = json.loads((output / "magi2_conversion_manifest.json").read_text())
    assert manifest["source"]["revision"] == SOURCE_REVISION
    assert manifest["component_directory_mapping"] == COMPONENT_DIRECTORY_MAPPING
    assert manifest["skipped_checkpoint_keys"]["vae/checkpoint.ckpt"] == list(
        TURBO_VAE_SKIPPED_KEYS
    )
    assert len(TURBO_VAE_SKIPPED_KEYS) == 8


def test_magi2_converter_rejects_an_incomplete_snapshot(tmp_path: Path) -> None:
    """Reject a source tree before creating a partial output repository."""
    source = tmp_path / "official"
    source.mkdir()
    output = tmp_path / "fastvideo"

    try:
        convert_checkpoint_layout(source, output)
    except FileNotFoundError as error:
        assert "source snapshot is incomplete" in str(error)
    else:
        raise AssertionError("An incomplete MAGI-2 snapshot was accepted")
    assert not output.exists()
