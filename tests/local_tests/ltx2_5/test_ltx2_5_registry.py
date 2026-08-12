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
