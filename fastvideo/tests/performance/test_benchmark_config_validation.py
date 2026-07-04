# SPDX-License-Identifier: Apache-2.0

import pytest

from fastvideo.tests.performance.test_inference_performance import (
    _benchmark_display_id,
    _config_identity_metadata,
    _is_v2_config,
    _validate_benchmark_config,
)


def test_v1_benchmark_config_without_schema_version_validates():
    cfg = {
        "benchmark_id": "legacy-benchmark",
    }

    _validate_benchmark_config(cfg, "legacy.json")

    assert _is_v2_config(cfg) is False
    assert _config_identity_metadata(cfg) == {}
    assert _benchmark_display_id(cfg) == "legacy-benchmark"


def test_v2_benchmark_config_identity_validates_and_is_preserved():
    cfg = {
        "benchmark_id": "wan-t2v-1.3b-2gpu",
        "config_schema_version": 2,
        "workload_id": "wan-t2v-1.3b",
        "variant_id": "canonical",
        "benchmark_version": 1,
    }

    _validate_benchmark_config(cfg, "wan.json")

    assert _is_v2_config(cfg) is True
    assert _config_identity_metadata(cfg) == {
        "config_schema_version": 2,
        "workload_id": "wan-t2v-1.3b",
        "variant_id": "canonical",
        "benchmark_version": 1,
    }


def test_v2_benchmark_config_missing_identity_fields_fails_clearly():
    cfg = {
        "benchmark_id": "wan-t2v-1.3b-2gpu",
        "config_schema_version": 2,
        "workload_id": "wan-t2v-1.3b",
    }

    expected = "wan.json: missing required v2 identity fields: variant_id, benchmark_version"
    with pytest.raises(ValueError, match=expected):
        _validate_benchmark_config(cfg, "wan.json")


def test_partial_v2_identity_requires_schema_version():
    cfg = {
        "benchmark_id": "wan-t2v-1.3b-2gpu",
        "workload_id": "wan-t2v-1.3b",
        "variant_id": "canonical",
        "benchmark_version": 1,
    }

    expected = "wan.json: v2 benchmark identity fields require config_schema_version=2"
    with pytest.raises(ValueError, match=expected):
        _validate_benchmark_config(cfg, "wan.json")


def test_optional_v2_metadata_fields_must_be_objects():
    cfg = {
        "benchmark_id": "wan-t2v-1.3b-2gpu",
        "config_schema_version": 2,
        "workload_id": "wan-t2v-1.3b",
        "variant_id": "canonical",
        "benchmark_version": 1,
        "quality_metadata": ["not", "an", "object"],
    }

    expected = "wan.json: optional v2 metadata field 'quality_metadata' must be an object"
    with pytest.raises(ValueError, match=expected):
        _validate_benchmark_config(cfg, "wan.json")
