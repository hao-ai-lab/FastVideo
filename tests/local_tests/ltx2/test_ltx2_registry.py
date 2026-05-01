# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest

pytest.importorskip("torchvision")


from fastvideo.api.sampling_param import SamplingParam
from fastvideo.registry import (
    get_pipeline_config_cls_from_name,
    get_sampling_param_cls_for_name,
)
from fastvideo.pipelines.basic.ltx2.pipeline_configs import LTX2T2VConfig


@pytest.mark.parametrize(
    ("model_id", "expected_variant"),
    [
        ("Lightricks/LTX-2", "base"),
        ("FastVideo/LTX2-base", "base"),
        ("FastVideo/LTX2-Distilled-Diffusers", "distilled"),
    ],
)
def test_ltx2_sampling_registry_exact_ids(
    model_id: str,
    expected_variant: str,
) -> None:
    # All sampling_param_cls are None after profile migration.
    assert get_sampling_param_cls_for_name(model_id) is None

    # Profile-based path should return correct defaults.
    sp = SamplingParam.from_pretrained(model_id)
    if expected_variant == "base":
        assert sp.num_inference_steps == 40
        assert sp.height == 512
        assert sp.width == 768
    else:
        assert sp.num_inference_steps == 8
        assert sp.height == 1024
        assert sp.width == 1536


@pytest.mark.parametrize(
    "model_id",
    [
        "Lightricks/LTX-2",
        "FastVideo/LTX2-base",
        "FastVideo/LTX2-Distilled-Diffusers",
    ],
)
def test_ltx2_pipeline_registry_exact_ids(model_id: str) -> None:
    resolved_cls = get_pipeline_config_cls_from_name(model_id)
    assert resolved_cls is LTX2T2VConfig


def _write_minimal_diffusers_repo(
    model_dir: Path, class_name: str
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "transformer").mkdir(exist_ok=True)
    (model_dir / "vae").mkdir(exist_ok=True)
    with (model_dir / "model_index.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "_class_name": class_name,
                "_diffusers_version": "0.33.0.dev0",
                "transformer": [
                    "diffusers",
                    "LTX2Transformer3DModel",
                ],
                "vae": ["diffusers", "CausalVideoAutoencoder"],
            },
            f,
        )


def test_ltx2_ambiguous_local_path_has_no_sampling_fallback(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "converted" / "ltx2_diffusers"
    _write_minimal_diffusers_repo(model_dir, "LTX2Pipeline")
    resolved_cls = get_sampling_param_cls_for_name(str(model_dir))
    assert resolved_cls is None


def test_ltx2_ambiguous_local_path_resolves_via_detector(
    tmp_path: Path,
) -> None:
    # Both base and distilled share LTX2T2VConfig, so the
    # "ltx2" detector correctly resolves ambiguous local paths.
    model_dir = tmp_path / "converted" / "ltx2_diffusers"
    _write_minimal_diffusers_repo(model_dir, "LTX2Pipeline")
    resolved_cls = get_pipeline_config_cls_from_name(str(model_dir))
    assert resolved_cls is LTX2T2VConfig
