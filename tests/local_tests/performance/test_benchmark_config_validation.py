# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest

from fastvideo.tests.performance.test_inference_performance import (
    _benchmark_display_id,
    _config_identity_metadata,
    _is_v2_config,
    _validate_benchmark_config,
)


def _v2_config():
    return {
        "benchmark_id": "wan-t2v-1.3b-2gpu",
        "config_schema_version": 2,
        "workload_id": "wan-t2v",
        "variant_id": "1.3b-sp2",
        "benchmark_version": 2,
    }


def test_v1_benchmark_config_without_schema_version_validates():
    cfg = {
        "benchmark_id": "legacy-benchmark",
    }

    _validate_benchmark_config(cfg, "legacy.json")

    assert _is_v2_config(cfg) is False
    assert _config_identity_metadata(cfg) == {}
    assert _benchmark_display_id(cfg) == "legacy-benchmark"


def test_v2_benchmark_config_identity_validates_and_is_preserved():
    cfg = _v2_config()
    cfg["quality_metadata"] = {"some": "data"}

    _validate_benchmark_config(cfg, "wan.json")

    assert _is_v2_config(cfg) is True
    assert _config_identity_metadata(cfg) == {
        "config_schema_version": 2,
        "workload_id": "wan-t2v",
        "variant_id": "1.3b-sp2",
        "benchmark_version": 2,
        "quality_metadata": {
            "some": "data"
        },
    }


def test_v2_benchmark_config_missing_identity_fields_fails_clearly():
    cfg = _v2_config()
    del cfg["variant_id"]
    del cfg["benchmark_version"]

    expected = "wan.json: missing required v2 identity fields: variant_id, benchmark_version"
    with pytest.raises(ValueError, match=expected):
        _validate_benchmark_config(cfg, "wan.json")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("workload_id", {}),
        ("workload_id", ""),
        ("workload_id", "   "),
        ("variant_id", []),
        ("variant_id", ""),
        ("variant_id", "   "),
    ],
)
def test_v2_benchmark_config_rejects_invalid_string_identity_values(field, value):
    cfg = _v2_config()
    cfg[field] = value

    expected = f"wan.json: v2 identity field {field!r} must be a non-empty string"
    with pytest.raises(ValueError, match=expected):
        _validate_benchmark_config(cfg, "wan.json")


@pytest.mark.parametrize("value", [None, "1", 1.5, True])
def test_v2_benchmark_config_rejects_invalid_benchmark_version_values(value):
    cfg = _v2_config()
    cfg["benchmark_version"] = value

    expected = "wan.json: v2 identity field 'benchmark_version' must be an integer"
    with pytest.raises(ValueError, match=expected):
        _validate_benchmark_config(cfg, "wan.json")


def test_partial_v2_identity_requires_schema_version():
    cfg = {
        "benchmark_id": "wan-t2v-1.3b-2gpu",
        "workload_id": "wan-t2v",
        "variant_id": "1.3b-sp2",
        "benchmark_version": 2,
    }

    expected = "wan.json: v2 benchmark identity fields require config_schema_version=2"
    with pytest.raises(ValueError, match=expected):
        _validate_benchmark_config(cfg, "wan.json")


def test_optional_v2_metadata_fields_must_be_objects():
    cfg = _v2_config()
    cfg["quality_metadata"] = ["not", "an", "object"]

    expected = "wan.json: optional v2 metadata field 'quality_metadata' must be an object"
    with pytest.raises(ValueError, match=expected):
        _validate_benchmark_config(cfg, "wan.json")


def test_regression_thresholds_validate_without_forcing_v2_schema():
    cfg = {
        "benchmark_id": "legacy-benchmark",
        "regression_thresholds": {
            "latency": {
                "threshold_percent": 0.1
            }
        },
    }

    _validate_benchmark_config(cfg, "legacy.json")

    cfg["regression_thresholds"] = []
    with pytest.raises(ValueError, match="benchmark config field 'regression_thresholds' must be an object"):
        _validate_benchmark_config(cfg, "legacy.json")


def test_gpu_type_gate_skips_non_matching_device():
    from fastvideo.tests.performance.test_inference_performance import (
        _gpu_type_skip_reason,
    )

    cfg = {"benchmark_id": "wan-t2v-1.3b-1gpu-gb10"}
    run_config = {"gpu_types": ["GB10"]}

    # Matching device (substring) runs; non-matching devices skip.
    assert _gpu_type_skip_reason(cfg, run_config, "NVIDIA GB10") is None
    assert _gpu_type_skip_reason(cfg, run_config, "NVIDIA H100 80GB HBM3") is not None
    assert _gpu_type_skip_reason(cfg, run_config, "NVIDIA L40S") is not None


def test_gpu_type_gate_absent_field_runs_everywhere():
    from fastvideo.tests.performance.test_inference_performance import (
        _gpu_type_skip_reason,
    )

    cfg = {"benchmark_id": "wan-t2v-1.3b-2gpu"}
    # No gpu_types key, and an empty list, both run on any device.
    assert _gpu_type_skip_reason(cfg, {}, "NVIDIA H100 80GB HBM3") is None
    assert _gpu_type_skip_reason(cfg, {"gpu_types": []}, "NVIDIA H100") is None


def test_gpu_types_must_be_list_of_nonempty_strings():
    # A bare string (would iterate characters), non-string entries, and empty
    # strings are all rejected at config-discovery time.
    for bad in ("GB10", ["GB10", 3], [""], [None]):
        cfg = {"benchmark_id": "x", "run_config": {"gpu_types": bad}}
        with pytest.raises(ValueError, match="gpu_types"):
            _validate_benchmark_config(cfg, "x.json")

    # A valid list, an absent field, and an empty list all pass.
    cfg = {
        "benchmark_id": "x",
        "run_config": {
            "gpu_types": ["GB10"]
        },
        "thresholds": {
            "GB10": {}, "default": {}
        },
    }
    _validate_benchmark_config(cfg, "x.json")
    _validate_benchmark_config({"benchmark_id": "x"}, "x.json")
    _validate_benchmark_config(
        {"benchmark_id": "x", "run_config": {"gpu_types": []}}, "x.json")


def test_gpu_types_require_matching_thresholds_and_default():
    cfg = {
        "benchmark_id": "x",
        "run_config": {
            "gpu_types": ["GB10"]
        },
        "thresholds": {
            "GB10": {}, "default": {}
        },
    }
    _validate_benchmark_config(cfg, "x.json")

    mismatched = {
        "benchmark_id": "x",
        "run_config": {
            "gpu_types": ["GB10", "DGX"]
        },
        "thresholds": {
            "GB10": {}, "default": {}
        },
    }
    with pytest.raises(ValueError, match="missing thresholds keys: DGX"):
        _validate_benchmark_config(mismatched, "x.json")

    no_default = {
        "benchmark_id": "x",
        "run_config": {
            "gpu_types": ["GB10"]
        },
        "thresholds": {
            "GB10": {}
        },
    }
    with pytest.raises(ValueError, match="requires a 'default' thresholds block"):
        _validate_benchmark_config(no_default, "x.json")


def test_shipped_gb10_config_gates_and_has_matching_thresholds_key():
    from fastvideo.tests.performance.test_inference_performance import (
        _BENCHMARKS_DIR,
        _gpu_type_skip_reason,
    )

    path = os.path.join(_BENCHMARKS_DIR, "wan-t2v-1.3b-gb10-1gpu.json")
    with open(path) as f:
        cfg = json.load(f)

    assert _gpu_type_skip_reason(cfg, cfg["run_config"], "NVIDIA GB10") is None
    assert "GB10" in cfg["thresholds"]
