# SPDX-License-Identifier: Apache-2.0
import pytest

from fastvideo.performance.cohort import (
    cohort_descriptor,
    cohort_key,
    cohort_schema,
    comparison_identity_filters,
    gpu_key,
)


def _v2_record(**overrides):
    record = {
        "result_schema_version": 2,
        "model_id": "wan-t2v-1.3b-2gpu",
        "gpu_type": "NVIDIA L40S",
        "workload_id": "wan-t2v",
        "variant_id": "1.3b-sp2",
        "benchmark_version": 2,
        "recipe_fingerprint": "recipe-a",
        "hardware_profile_id": "hw-l40s-sp2",
        "software_profile_id": "sw-cu126",
        "hardware_profile": {
            "gpu_count": 2,
            "gpus": [{"name": "NVIDIA L40S", "memory_gb": 48}] * 2,
            "interconnect": "full_nvlink",
        },
        "software_profile": {
            "python": "3.12",
            "pytorch": "2.7",
            "cuda": "12.6",
        },
        "environment_metadata": {"platform": {"machine": "x86_64"}},
    }
    record.update(overrides)
    return record


def test_v2_cohort_key_ignores_display_metadata_renames():
    original = _v2_record()
    renamed = _v2_record(model_id="new-display-name", gpu_type="NVIDIA L40S new label")

    assert cohort_key(original) == cohort_key(renamed)
    assert gpu_key(original) == gpu_key(renamed)


def test_legacy_and_v2_cohorts_never_share_a_key():
    legacy = {"model_id": "wan-t2v-1.3b-2gpu", "gpu_type": "NVIDIA L40S"}

    assert cohort_schema(legacy) == "legacy"
    assert cohort_key(legacy) != cohort_key(_v2_record())


def test_partial_v2_identity_is_explicitly_invalid():
    record = {
        "result_schema_version": 2,
        "model_id": "wan",
        "gpu_type": "NVIDIA L40S",
        "workload_id": "wan-t2v",
    }

    assert cohort_schema(record) == "invalid_v2"
    with pytest.raises(ValueError, match="variant_id"):
        comparison_identity_filters(record)


def test_cohort_descriptor_exposes_readable_labels_and_raw_ids():
    descriptor = cohort_descriptor(_v2_record())

    assert descriptor["title"] == "wan-t2v · 1.3b-sp2 · v2"
    assert descriptor["gpu_label"] == "2× NVIDIA L40S · 48 GB · full nvlink"
    assert descriptor["hardware_label"].endswith("· x86_64")
    assert descriptor["software_label"] == "CUDA 12.6 · PyTorch 2.7 · Python 3.12"
    assert descriptor["raw_ids"] == {
        "hardware_profile_id": "hw-l40s-sp2",
        "software_profile_id": "sw-cu126",
        "recipe_fingerprint": "recipe-a",
    }
