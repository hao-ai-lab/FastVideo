# SPDX-License-Identifier: Apache-2.0
import json
import importlib
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("torchvision")


def _bootstrap_fastvideo_namespace() -> None:
    """Avoid importing fastvideo/__init__.py during local registry tests."""
    if "fastvideo" in sys.modules:
        return

    repo_root = Path(__file__).resolve().parents[2]
    package_dir = repo_root / "fastvideo"
    fastvideo_pkg = types.ModuleType("fastvideo")
    fastvideo_pkg.__path__ = [str(package_dir)]  # type: ignore[attr-defined]
    fastvideo_pkg.__file__ = str(package_dir / "__init__.py")
    sys.modules["fastvideo"] = fastvideo_pkg


def _get_registry_test_symbols() -> tuple[type, type, type, object, object]:
    _bootstrap_fastvideo_namespace()

    pipeline_module = importlib.import_module("fastvideo.configs.pipelines.ltx2")
    sample_module = importlib.import_module("fastvideo.configs.sample.ltx2")
    registry_module = importlib.import_module("fastvideo.registry")

    return (
        pipeline_module.LTX2T2VConfig,
        sample_module.LTX2BaseSamplingParam,
        sample_module.LTX2DistilledSamplingParam,
        registry_module.get_pipeline_config_cls_from_name,
        registry_module.get_sampling_param_cls_for_name,
    )


@pytest.mark.parametrize(
    ("model_id", "expected_variant"),
    [
        ("Lightricks/LTX-2", "base"),
        ("FastVideo/LTX2-base", "base"),
        ("FastVideo/LTX2-Distilled-Diffusers", "distilled"),
    ],
)
def test_ltx2_sampling_registry_exact_ids(model_id: str,
                                          expected_variant: str) -> None:
    _, base_cls, distilled_cls, _, get_sampling_param_cls_for_name = _get_registry_test_symbols()
    expected_cls = base_cls if expected_variant == "base" else distilled_cls
    resolved_cls = get_sampling_param_cls_for_name(model_id)
    assert resolved_cls is expected_cls


@pytest.mark.parametrize(
    "model_id",
    [
        "Lightricks/LTX-2",
        "FastVideo/LTX2-base",
        "FastVideo/LTX2-Distilled-Diffusers",
    ],
)
def test_ltx2_pipeline_registry_exact_ids(model_id: str) -> None:
    pipeline_cls, _, _, get_pipeline_config_cls_from_name, _ = _get_registry_test_symbols()
    resolved_cls = get_pipeline_config_cls_from_name(model_id)
    assert resolved_cls is pipeline_cls


def _write_minimal_diffusers_repo(model_dir: Path, class_name: str) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "transformer").mkdir(exist_ok=True)
    (model_dir / "vae").mkdir(exist_ok=True)
    with (model_dir / "model_index.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "_class_name": class_name,
                "_diffusers_version": "0.33.0.dev0",
                "transformer": ["diffusers", "LTX2Transformer3DModel"],
                "vae": ["diffusers", "CausalVideoAutoencoder"],
            },
            f,
        )


def test_ltx2_ambiguous_local_path_has_no_sampling_fallback(
        tmp_path: Path) -> None:
    _, _, _, _, get_sampling_param_cls_for_name = _get_registry_test_symbols()
    # Simulate a user-local converted LTX2 path that might be either base or
    # distilled. Registry must not assume a variant for local converted paths.
    model_dir = tmp_path / "converted" / "ltx2_diffusers"
    _write_minimal_diffusers_repo(model_dir, "LTX2Pipeline")

    resolved_cls = get_sampling_param_cls_for_name(str(model_dir))
    assert resolved_cls is None


def test_ltx2_ambiguous_local_path_has_no_pipeline_mapping(
        tmp_path: Path) -> None:
    _, _, _, get_pipeline_config_cls_from_name, _ = _get_registry_test_symbols()
    model_dir = tmp_path / "converted" / "ltx2_diffusers"
    _write_minimal_diffusers_repo(model_dir, "LTX2Pipeline")

    with pytest.raises(ValueError,
                       match="No match found for pipeline .*check the pipeline name or path"):
        get_pipeline_config_cls_from_name(str(model_dir))
