# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import fastvideo.pipelines as pipelines
import fastvideo.utils as utils
from fastvideo.pipelines.basic.minimax_h3 import (
    MiniMaxH3ModularPipeline,
    MiniMaxH3Ref2VAModularPipeline,
)


def test_maybe_download_model_downloads_only_selected_scheduler(tmp_path: Path, monkeypatch) -> None:
    captured: dict = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        scheduler_dir = tmp_path / "scheduler"
        scheduler_dir.mkdir()
        (scheduler_dir / "scheduler_config.json").write_text("{}")
        return str(tmp_path)

    monkeypatch.setattr(utils, "snapshot_download", fake_snapshot_download)

    model_path = Path(
        utils.maybe_download_model(
            "MiniMaxAI/MiniMax-H3",
            local_dir=str(tmp_path),
            allow_patterns=["scheduler/**"],
        ))

    assert captured["allow_patterns"] == ["scheduler/**"]
    assert (model_path / "scheduler" / "scheduler_config.json").is_file()
    assert not (model_path / "transformer").exists()
    assert not (model_path / "transformer_ref").exists()


def test_minimax_h3_selects_only_its_transformer_partition() -> None:
    standard_dirs = set(MiniMaxH3ModularPipeline.get_hf_download_component_dirs())
    ref2va_dirs = set(MiniMaxH3Ref2VAModularPipeline.get_hf_download_component_dirs())

    assert "transformer" in standard_dirs
    assert "transformer_ref" not in standard_dirs
    assert "transformer_ref" in ref2va_dirs
    assert "transformer" not in ref2va_dirs


def test_build_pipeline_forwards_selected_component_patterns(monkeypatch) -> None:
    captured: dict = {}

    class FakeMiniMaxPipeline:

        @classmethod
        def get_hf_download_component_dirs(cls):
            return ("scheduler", )

        def __init__(self, model_path, fastvideo_args):
            captured["pipeline_model_path"] = model_path

    monkeypatch.setattr(
        pipelines,
        "get_model_info",
        lambda **kwargs: SimpleNamespace(pipeline_cls=FakeMiniMaxPipeline),
    )

    def fake_download(model_path, **kwargs):
        captured.update(kwargs)
        return "/tmp/minimax-h3"

    monkeypatch.setattr(pipelines, "maybe_download_model", fake_download)
    args = SimpleNamespace(
        model_path="MiniMaxAI/MiniMax-H3",
        revision="test-revision",
        workload_type=None,
        override_pipeline_cls_name=None,
    )

    pipelines.build_pipeline(args)

    assert captured["allow_patterns"] == [
        "model_index.json",
        "modular_model_index.json",
        "scheduler/**",
    ]
    assert captured["revision"] == "test-revision"
    assert captured["pipeline_model_path"] == "/tmp/minimax-h3"


def test_model_index_supports_umbrella_repo_paths(tmp_path: Path, monkeypatch) -> None:
    captured: dict = {}

    def fake_hf_hub_download(**kwargs):
        captured.update(kwargs)
        manifest_path = tmp_path / kwargs["filename"]
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({
            "_class_name": "MiniMaxH3ModularPipeline",
            "_diffusers_version": "0.35.0",
        }))
        return str(manifest_path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

    config = utils.maybe_download_model_index("org/repo/minimax-h3", revision="test-revision")

    assert captured["repo_id"] == "org/repo"
    assert captured["filename"] == "minimax-h3/model_index.json"
    assert captured["revision"] == "test-revision"
    assert config["pipeline_name"] == "MiniMaxH3ModularPipeline"
