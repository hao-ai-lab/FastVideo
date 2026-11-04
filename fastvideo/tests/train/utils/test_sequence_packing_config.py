# SPDX-License-Identifier: Apache-2.0
"""Strict config parsing for cross-sample sequence packing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

import fastvideo.dataset
from fastvideo.train.utils.dataloader import build_parquet_t2v_train_dataloader
from fastvideo.train.utils.config import load_run_config
from fastvideo.train.utils.training_config import CrossSampleSequencePackingConfig, DataConfig


def _config(data: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "models": {
            "student": {
                "_target_": "fastvideo.train.models.minimax_h3.minimax_h3.MiniMaxH3Model",
                "init_from": "fake/model",
            }
        },
        "method": {
            "_target_": "fastvideo.train.methods.distribution_matching.dmd2.DMD2Method",
        },
        "training": {
            "data": data or {},
        },
    }


def _write(tmp_path: Path, config: dict[str, Any]) -> str:
    path = tmp_path / "packing.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return str(path)


def test_sequence_packing_defaults_off(tmp_path: Path) -> None:
    config = load_run_config(_write(tmp_path, _config()))

    assert config.training.data.cross_sample_sequence_packing.enabled is False
    assert config.training.data.cross_sample_sequence_packing.max_joint_tokens is None
    assert config.training.data.cross_sample_sequence_packing.max_samples_per_pack is None
    assert config.training.data.fixed_sequence_capacities is None


def test_sequence_packing_parses_complete_opt_in(tmp_path: Path) -> None:
    config = load_run_config(
        _write(
            tmp_path,
            _config({
                "preprocessed_data_type": "text_only",
                "train_batch_size": 1,
                "native_shape_bucketing": True,
                "cross_sample_sequence_packing": {
                    "enabled": True,
                    "max_joint_tokens": 220_000,
                    "max_samples_per_pack": 4,
                },
            }),
        ))

    packing = config.training.data.cross_sample_sequence_packing
    assert packing.enabled is True
    assert packing.max_joint_tokens == 220_000
    assert packing.max_samples_per_pack == 4


@pytest.mark.parametrize(
    "packing,match",
    [
        (True, "Expected mapping"),
        (None, "Expected mapping"),
        ({"enabled": 1, "max_joint_tokens": 10}, "enabled"),
        ({"enabled": True, "max_joint_tokens": "10"}, "positive YAML integer"),
        ({"enabled": True, "max_joint_tokens": 10.0}, "positive YAML integer"),
        ({"enabled": True, "max_joint_tokens": 10, "max_samples_per_pack": False},
         "positive YAML integer"),
        ({"enabled": False, "typo": 10}, "Unknown key"),
    ],
)
def test_sequence_packing_nested_values_are_strict(
    tmp_path: Path,
    packing: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        load_run_config(_write(tmp_path, _config({"cross_sample_sequence_packing": packing})))


@pytest.mark.parametrize(
    "data,match",
    [
        ({
            "native_shape_bucketing": True,
            "cross_sample_sequence_packing": {"enabled": True},
        }, "max_joint_tokens is required"),
        ({
            "cross_sample_sequence_packing": {"enabled": True, "max_joint_tokens": 10},
        }, "native_shape_bucketing"),
        ({
            "native_shape_bucketing": True,
            "train_batch_size": 2,
            "cross_sample_sequence_packing": {"enabled": True, "max_joint_tokens": 10},
        }, "train_batch_size"),
        ({
            "native_shape_bucketing": True,
            "train_batch_size": "1",
            "cross_sample_sequence_packing": {"enabled": True, "max_joint_tokens": 10},
        }, "YAML integer 1"),
        ({
            "preprocessed_data_type": "ref2va",
            "native_shape_bucketing": True,
            "cross_sample_sequence_packing": {"enabled": True, "max_joint_tokens": 10},
        }, "only text_only and t2va"),
    ],
)
def test_sequence_packing_fails_closed_on_unsupported_data_contract(
    tmp_path: Path,
    data: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        load_run_config(_write(tmp_path, _config(data)))


def test_t2v_dataloader_plumbs_packing_and_accumulation_window(monkeypatch) -> None:
    observed: dict[str, Any] = {}

    def fake_builder(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return object(), "loader"

    monkeypatch.setattr(fastvideo.dataset, "build_parquet_map_style_dataloader", fake_builder)
    data = DataConfig(
        data_path="/fake/data",
        train_batch_size=1,
        native_shape_bucketing=True,
        fixed_sequence_capacities=[262144],
        cross_sample_sequence_packing=CrossSampleSequencePackingConfig(
            enabled=True,
            max_joint_tokens=220_000,
            max_samples_per_pack=4,
        ),
    )

    loader = build_parquet_t2v_train_dataloader(
        data,
        text_len=None,
        parquet_schema=object(),
        max_presentation_tokens=4096,
        gradient_accumulation_steps=4,
    )

    assert loader == "loader"
    assert observed["kwargs"]["sequence_packing_enabled"] is True
    assert observed["kwargs"]["fixed_sequence_capacities"] == [262144]
    assert observed["kwargs"]["sequence_packing_max_joint_tokens"] == 220_000
    assert observed["kwargs"]["sequence_packing_max_samples_per_pack"] == 4
    assert observed["kwargs"]["gradient_accumulation_steps"] == 4


def test_packed_t2v_dataloader_requires_accumulation_window() -> None:
    data = DataConfig(
        native_shape_bucketing=True,
        cross_sample_sequence_packing=CrossSampleSequencePackingConfig(
            enabled=True,
            max_joint_tokens=220_000,
        ),
    )

    with pytest.raises(ValueError, match="requires gradient_accumulation_steps"):
        build_parquet_t2v_train_dataloader(
            data,
            text_len=None,
            parquet_schema=object(),
        )


@pytest.mark.parametrize("data_type", ["text_only", "t2va", "ref2va"])
def test_fixed_sequence_capacities_parse_without_enabling_cross_sample_packing(tmp_path, data_type) -> None:
    config = load_run_config(_write(tmp_path, _config({
        "preprocessed_data_type": data_type,
        "native_shape_bucketing": True,
        "fixed_sequence_capacities": [161280, 131072, 161280],
    })))
    assert config.training.data.fixed_sequence_capacities == [131072, 161280]
    assert config.training.data.cross_sample_sequence_packing.enabled is False
    assert config.training.data.train_batch_size == 1


@pytest.mark.parametrize("values", [[], [0], [-256], [255], [True], [256.0], ["256"], "256", 256, {}])
def test_fixed_sequence_capacities_strict_yaml_list_validation(tmp_path, values) -> None:
    with pytest.raises(ValueError, match="fixed_sequence_capacities"):
        load_run_config(_write(tmp_path, _config({
            "preprocessed_data_type": "ref2va",
            "native_shape_bucketing": True,
            "fixed_sequence_capacities": values,
        })))


@pytest.mark.parametrize(
    "override,match",
    [
        ({"native_shape_bucketing": False}, "native_shape_bucketing"),
        ({"train_batch_size": 2}, "YAML integer 1"),
        ({"train_batch_size": 0}, "YAML integer 1"),
        ({"train_batch_size": True}, "YAML integer 1"),
        ({"train_batch_size": 1.0}, "YAML integer 1"),
        ({"train_batch_size": "1"}, "YAML integer 1"),
        ({"train_batch_size": None}, "YAML integer 1"),
        ({"preprocessed_data_type": "t2v"}, "only text_only, t2va and ref2va"),
        ({"cross_sample_sequence_packing": {"enabled": True, "max_joint_tokens": 513}},
         "exceeds largest fixed_sequence_capacities"),
    ],
)
def test_fixed_sequence_capacities_reject_unsupported_contracts(tmp_path, override, match) -> None:
    data = {
        "preprocessed_data_type": "t2va",
        "native_shape_bucketing": True,
        "fixed_sequence_capacities": [256, 512],
        **override,
    }
    with pytest.raises(ValueError, match=match):
        load_run_config(_write(tmp_path, _config(data)))


def test_fixed_sequence_capacities_allow_explicit_dmd_cross_sample_packing(tmp_path) -> None:
    config = load_run_config(_write(tmp_path, _config({
        "preprocessed_data_type": "t2va",
        "native_shape_bucketing": True,
        "fixed_sequence_capacities": [256, 512],
        "cross_sample_sequence_packing": {"enabled": True, "max_joint_tokens": 512},
    })))
    assert config.training.data.cross_sample_sequence_packing.enabled is True
    assert config.training.data.cross_sample_sequence_packing.max_joint_tokens == 512
    assert config.training.data.fixed_sequence_capacities == [256, 512]
