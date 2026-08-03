# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import ANY

import pytest

from fastvideo.pipelines import composed_pipeline_base
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase


class _ManifestPipeline(ComposedPipelineBase):
    _required_config_modules = ["transformer"]

    def _load_config(self, model_path: str) -> dict[str, Any]:
        del model_path
        return dict(self.manifest)

    def create_pipeline_stages(self, fastvideo_args: Any) -> None:
        del fastvideo_args


class _MiniMaxH3RefLoaderPipeline(_ManifestPipeline):
    _extra_config_module_map = {"transformer": "transformer_ref"}


class _MiniMaxH3FLLoaderPipeline(_ManifestPipeline):
    pass


def _make_pipeline(
    pipeline_cls: type[_ManifestPipeline],
    model_path: Path,
    manifest: dict[str, Any],
) -> _ManifestPipeline:
    pipeline = pipeline_cls.__new__(pipeline_cls)
    pipeline.model_path = str(model_path)
    pipeline.manifest = manifest
    return pipeline


def _manifest(*, include_transformer_ref: bool = True) -> dict[str, Any]:
    manifest = {
        "_class_name": "MiniMaxH3ModularPipeline",
        "_diffusers_version": "0.36.0.dev0",
        "transformer": ["standard-library", "MiniMaxH3Transformer3DModel"],
        "scheduler": ["diffusers", "MiniMaxH3Scheduler"],
    }
    if include_transformer_ref:
        manifest["transformer_ref"] = ["reference-library", "MiniMaxH3Transformer3DModel"]
    return manifest


def _record_component_loads(monkeypatch: pytest.MonkeyPatch) -> tuple[list[dict[str, Any]], object]:
    calls: list[dict[str, Any]] = []
    loaded_transformer = object()

    def load_module(**kwargs: Any) -> object:
        calls.append(kwargs)
        return loaded_transformer

    monkeypatch.setattr(composed_pipeline_base.PipelineComponentLoader, "load_module", load_module)
    return calls, loaded_transformer


def test_ref_loader_resolves_transformer_ref_as_logical_transformer(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    calls, loaded_transformer = _record_component_loads(monkeypatch)
    pipeline = _make_pipeline(_MiniMaxH3RefLoaderPipeline, tmp_path, _manifest())

    modules = pipeline.load_modules(SimpleNamespace())

    assert modules == {"transformer": loaded_transformer}
    assert calls == [{
        "module_name": "transformer",
        "component_model_path": str(tmp_path / "transformer_ref"),
        "transformers_or_diffusers": "reference-library",
        "fastvideo_args": ANY,
    }]


def test_ref_loader_requires_transformer_ref(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls, _ = _record_component_loads(monkeypatch)
    pipeline = _make_pipeline(
        _MiniMaxH3RefLoaderPipeline,
        tmp_path,
        _manifest(include_transformer_ref=False),
    )

    with pytest.raises(ValueError, match="Required source module 'transformer_ref'.*logical module 'transformer'"):
        pipeline.load_modules(SimpleNamespace())

    assert calls == []


def test_fl_loader_uses_only_standard_transformer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls, loaded_transformer = _record_component_loads(monkeypatch)
    pipeline = _make_pipeline(_MiniMaxH3FLLoaderPipeline, tmp_path, _manifest())

    modules = pipeline.load_modules(SimpleNamespace())

    assert modules == {"transformer": loaded_transformer}
    assert calls == [{
        "module_name": "transformer",
        "component_model_path": str(tmp_path / "transformer"),
        "transformers_or_diffusers": "standard-library",
        "fastvideo_args": ANY,
    }]
