# SPDX-License-Identifier: Apache-2.0
"""Registry coverage for every official MMAudio model variant."""

import json
from pathlib import Path

import pytest

from fastvideo.configs.pipelines.mmaudio import (
    MMAudioLarge44kV2PipelineConfig,
    MMAudioLarge44kV2AConfig,
    MMAudioMedium44kV2AConfig,
    MMAudioSmall16kV2AConfig,
    MMAudioSmall44kV2AConfig,
    MMAudioV2AConfig,
)
from fastvideo.registry import get_default_preset, get_pipeline_config_cls_from_name


@pytest.mark.parametrize(
    ("variant", "config_cls"),
    [
        ("small-16k", MMAudioSmall16kV2AConfig),
        ("small-44k", MMAudioSmall44kV2AConfig),
        ("medium-44k", MMAudioMedium44kV2AConfig),
        ("large-44k", MMAudioLarge44kV2AConfig),
        ("large-44k-v2", MMAudioLarge44kV2PipelineConfig),
    ],
)
def test_mmaudio_variant_registry(variant: str, config_cls: type) -> None:
    model_path = f"FastVideo/MMAudio-{variant}-Diffusers"

    assert get_pipeline_config_cls_from_name(model_path) is config_cls
    assert get_default_preset(model_path) == f"mmaudio_{variant.replace('-', '_')}"


@pytest.mark.parametrize(
    ("directory_name", "config_cls", "preset"),
    [
        ("preprocess_44k", MMAudioV2AConfig, "mmaudio_large_44k_v2"),
        ("preprocess_16k", MMAudioSmall16kV2AConfig, "mmaudio_small_16k"),
    ],
)
def test_unversioned_mmaudio_preprocessor_paths_remain_compatible(
    tmp_path: Path,
    directory_name: str,
    config_cls: type,
    preset: str,
) -> None:
    model_path = tmp_path / "mmaudio" / directory_name
    model_path.mkdir(parents=True)
    (model_path / "model_index.json").write_text(
        json.dumps({
            "_class_name": "MMAudioPipeline",
            "_diffusers_version": "0.36.0",
            "_fastvideo_preprocessor_only": True,
        }),
        encoding="utf-8",
    )

    assert get_pipeline_config_cls_from_name(str(model_path)) is config_cls
    assert get_default_preset(str(model_path)) == preset
