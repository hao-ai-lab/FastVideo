# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest

from fastvideo.batching.admission import (
    AdmissionLimit,
    BatchAdmissionController,
    BatchingRule,
    load_batching_config,
)
from fastvideo.configs.pipelines.base import PipelineConfig


def test_admission_limit_rejects_batch_size_and_cost() -> None:
    limit = AdmissionLimit(max_batch_size=2, max_cost=10.0)

    assert limit.reject_reason(batch_size=3, batch_cost=1.0) == "config_cap:2"
    assert limit.reject_reason(batch_size=2, batch_cost=11.0) == "cost_budget:11>10"
    assert limit.reject_reason(batch_size=2, batch_cost=10.0) is None


def test_batching_rule_validates_unknown_keys() -> None:
    with pytest.raises(ValueError, match="did you mean 'max_batch_size'"):
        BatchingRule.from_dict(
            {
                "model_contains": "wan",
                "max_batch_siz": 2,
            },
            source="unit",
        )


@pytest.mark.parametrize(("value", "expected"), [(1, True), (0, False), (1.0, True), (0.0, False)])
def test_batching_rule_parses_numeric_bool_values(value, expected) -> None:
    rule = BatchingRule.from_dict(
        {
            "model_contains": "wan",
            "offload": value,
            "max_batch_size": 2,
        },
        source="unit",
    )

    assert rule.offload is expected


def test_load_batching_config_supports_mapping_form(tmp_path) -> None:
    path = tmp_path / "batching.json"
    path.write_text(
        '{"schema_version": 1, "wan|720x1280x81": {"max_batch_size": 3, "max_cost": 9}}',
        encoding="utf-8",
    )

    rules = load_batching_config(str(path))

    assert len(rules) == 1
    assert rules[0].model == "wan"
    assert rules[0].resolution == "720x1280x81"
    assert rules[0].max_batch_size == 3
    assert rules[0].max_cost == 9.0


def test_admission_controller_applies_user_and_config_caps(tmp_path, monkeypatch) -> None:
    path = tmp_path / "batching.json"
    path.write_text(
        '{"rules": [{"model_contains": "wan", "resolution": "720x1280x81", "max_batch_size": 3}]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(BatchAdmissionController, "_get_device_memory_gb", staticmethod(lambda gpu_id: 48.0))

    args = SimpleNamespace(
        batching_mode="dynamic",
        batching_max_size=4,
        batching_config=str(path),
        model_path="/models/wan",
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        pipeline_config=PipelineConfig(),
    )
    request = SimpleNamespace(height=720, width=1280, num_frames=81)

    controller = BatchAdmissionController(args)

    assert controller.enabled is True
    assert controller.max_admissible_batch_size(request) == 3
    assert controller.batch_is_full([request, request, request]) is True
