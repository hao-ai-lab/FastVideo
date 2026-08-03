# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.pipelines import composed_pipeline_base
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.utils import maybe_download_model_index, verify_model_config_and_directory


def _write_manifest(path: Path, filename: str, class_name: str, **components: Any) -> None:
    (path / filename).write_text(
        json.dumps({
            "_class_name": class_name,
            "_diffusers_version": "0.36.0",
            **components,
        }),
        encoding="utf-8",
    )


def test_local_modular_manifest_accepts_matching_component_subfolders(tmp_path: Path) -> None:
    (tmp_path / "transformer").mkdir()
    (tmp_path / "scheduler").mkdir()
    _write_manifest(
        tmp_path,
        "modular_model_index.json",
        "ModularPipeline",
        transformer=["diffusers", "Transformer", {"subfolder": "transformer"}],
        scheduler=["diffusers", "Scheduler", {"subfolder": "scheduler"}],
    )

    config = verify_model_config_and_directory(str(tmp_path))

    assert config["_class_name"] == "ModularPipeline"


def test_local_modular_manifest_uses_type_hint_to_detect_active_component(tmp_path: Path) -> None:
    (tmp_path / "transformer").mkdir()
    _write_manifest(
        tmp_path,
        "modular_model_index.json",
        "ModularPipeline",
        scheduler=[
            None,
            None,
            {
                "type_hint": ["diffusers", "Scheduler"],
                "subfolder": "scheduler",
            },
        ],
    )

    with pytest.raises(ValueError, match="missing the materialized scheduler/ subfolder"):
        verify_model_config_and_directory(str(tmp_path))


def test_local_modular_manifest_rejects_unloadable_component_subfolder(tmp_path: Path) -> None:
    (tmp_path / "transformer").mkdir()
    (tmp_path / "scheduler_config").mkdir()
    _write_manifest(
        tmp_path,
        "modular_model_index.json",
        "ModularPipeline",
        transformer=["diffusers", "Transformer", {"subfolder": "transformer"}],
        scheduler=["diffusers", "Scheduler", {"subfolder": "scheduler_config"}],
    )

    with pytest.raises(ValueError, match="requires component names and subfolders to match"):
        verify_model_config_and_directory(str(tmp_path))


def test_local_legacy_manifest_is_preferred_when_both_exist(tmp_path: Path) -> None:
    (tmp_path / "transformer").mkdir()
    _write_manifest(tmp_path, "model_index.json", "LegacyPipeline")
    _write_manifest(tmp_path, "modular_model_index.json", "ModularPipeline")

    config = verify_model_config_and_directory(str(tmp_path))

    assert config["_class_name"] == "LegacyPipeline"


def test_remote_manifest_download_falls_back_to_modular(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []

    def fake_download(*, repo_id: str, filename: str, local_dir: str) -> str:
        del repo_id, local_dir
        calls.append(filename)
        if filename == "model_index.json":
            raise FileNotFoundError(filename)
        manifest = tmp_path / filename
        _write_manifest(tmp_path, filename, "ModularPipeline")
        return str(manifest)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

    config = maybe_download_model_index("org/modular-model")

    assert calls == ["model_index.json", "modular_model_index.json"]
    assert config["pipeline_name"] == "ModularPipeline"


def test_explicit_pipeline_config_bypasses_registry(monkeypatch) -> None:
    explicit_config = PipelineConfig()

    def fail_registry_lookup(model_path: str):
        raise AssertionError(f"unexpected registry lookup for {model_path}")

    monkeypatch.setattr("fastvideo.registry.get_pipeline_config_cls_from_name", fail_registry_lookup)

    resolved = PipelineConfig.from_kwargs({
        "model_path": "/models/custom",
        "pipeline_config": explicit_config,
    })

    assert resolved is explicit_config


class _ExplicitConfigPipeline(ComposedPipelineBase):

    def __init__(
        self,
        model_path: str,
        fastvideo_args: Any,
        required_config_modules: list[str] | None = None,
        loaded_modules: dict[str, Any] | None = None,
    ) -> None:
        del model_path, required_config_modules, loaded_modules
        self.fastvideo_args = fastvideo_args

    def post_init(self) -> None:
        pass

    def create_pipeline_stages(self, fastvideo_args: Any) -> None:
        del fastvideo_args


def test_from_pretrained_forwards_explicit_pipeline_config(monkeypatch) -> None:
    explicit_config = PipelineConfig()
    captured: dict[str, Any] = {}

    def fake_from_kwargs(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(composed_pipeline_base.FastVideoArgs, "from_kwargs", fake_from_kwargs)

    pipeline = _ExplicitConfigPipeline.from_pretrained(
        "/models/custom",
        pipeline_config=explicit_config,
    )

    assert captured["pipeline_config"] is explicit_config
    assert pipeline.fastvideo_args is not None


class _ManifestLoaderPipeline(ComposedPipelineBase):
    _required_config_modules = ["transformer"]

    def _load_config(self, model_path: str) -> dict[str, Any]:
        del model_path
        return dict(self.manifest)

    def create_pipeline_stages(self, fastvideo_args: Any) -> None:
        del fastvideo_args


class _AliasedManifestLoaderPipeline(_ManifestLoaderPipeline):
    _extra_config_module_map = {"transformer": "transformer_ref"}


def _make_manifest_loader_pipeline(
    pipeline_cls: type[_ManifestLoaderPipeline],
    model_path: Path,
    manifest: dict[str, Any],
) -> _ManifestLoaderPipeline:
    pipeline = pipeline_cls.__new__(pipeline_cls)
    pipeline.model_path = str(model_path)
    pipeline.manifest = manifest
    return pipeline


@pytest.mark.parametrize(
    ("component_spec", "expected_library"),
    [
        (["legacy-library", "Transformer"], "legacy-library"),
        (
            [
                None,
                None,
                {
                    "type_hint": ["hint-library", "Transformer"],
                    "subfolder": "transformer",
                },
            ],
            "hint-library",
        ),
        (["fallback-library", "Transformer", {"subfolder": "transformer"}], "fallback-library"),
    ],
)
def test_composed_loader_accepts_legacy_and_modular_component_specs(
    tmp_path: Path,
    monkeypatch,
    component_spec: list[Any],
    expected_library: str,
) -> None:
    calls: list[dict[str, Any]] = []
    loaded = object()
    monkeypatch.setattr(
        composed_pipeline_base.PipelineComponentLoader,
        "load_module",
        lambda **kwargs: calls.append(kwargs) or loaded,
    )
    pipeline = _make_manifest_loader_pipeline(
        _ManifestLoaderPipeline,
        tmp_path,
        {
            "_class_name": "ManifestPipeline",
            "_diffusers_version": "0.36.0",
            "metadata_list": ["not", "a", "component", "spec"],
            "transformer": component_spec,
        },
    )

    modules = pipeline.load_modules(SimpleNamespace())

    assert modules == {"transformer": loaded}
    assert calls[0]["module_name"] == "transformer"
    assert calls[0]["component_model_path"] == str(tmp_path / "transformer")
    assert calls[0]["transformers_or_diffusers"] == expected_library


def test_composed_loader_alias_overrides_existing_logical_manifest_entry(tmp_path: Path, monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    loaded = object()
    monkeypatch.setattr(
        composed_pipeline_base.PipelineComponentLoader,
        "load_module",
        lambda **kwargs: calls.append(kwargs) or loaded,
    )
    pipeline = _make_manifest_loader_pipeline(
        _AliasedManifestLoaderPipeline,
        tmp_path,
        {
            "_class_name": "ManifestPipeline",
            "_diffusers_version": "0.36.0",
            "transformer": ["standard-library", "Transformer"],
            "transformer_ref": [
                None,
                None,
                {
                    "type_hint": ["reference-library", "Transformer"],
                    "subfolder": "transformer_ref",
                },
            ],
        },
    )

    modules = pipeline.load_modules(SimpleNamespace())

    assert modules == {"transformer": loaded}
    assert calls[0]["module_name"] == "transformer"
    assert calls[0]["component_model_path"] == str(tmp_path / "transformer_ref")
    assert calls[0]["transformers_or_diffusers"] == "reference-library"


def test_composed_loader_alias_requires_physical_manifest_entry_even_if_logical_exists(tmp_path: Path) -> None:
    pipeline = _make_manifest_loader_pipeline(
        _AliasedManifestLoaderPipeline,
        tmp_path,
        {
            "_class_name": "ManifestPipeline",
            "_diffusers_version": "0.36.0",
            "transformer": ["standard-library", "Transformer"],
        },
    )

    with pytest.raises(ValueError, match="Required source module 'transformer_ref'.*logical module 'transformer'"):
        pipeline.load_modules(SimpleNamespace())


@pytest.mark.parametrize(
    ("component_spec", "error"),
    [
        (["library", "Transformer", []], "Invalid Diffusers loading method"),
        (
            ["library", "Transformer", {"type_hint": ["library"]}],
            "Invalid Diffusers type_hint",
        ),
    ],
)
def test_composed_loader_rejects_malformed_modular_component_specs(
    tmp_path: Path,
    component_spec: list[Any],
    error: str,
) -> None:
    pipeline = _make_manifest_loader_pipeline(
        _ManifestLoaderPipeline,
        tmp_path,
        {
            "_class_name": "ManifestPipeline",
            "_diffusers_version": "0.36.0",
            "transformer": component_spec,
        },
    )

    with pytest.raises(ValueError, match=error):
        pipeline.load_modules(SimpleNamespace())
