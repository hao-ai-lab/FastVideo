# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import fastvideo.pipelines as pipelines
import fastvideo.utils as utils
import pytest
from huggingface_hub.utils import filter_repo_objects
from fastvideo.pipelines.basic.minimax_h3 import (
    MiniMaxH3ModularPipeline,
    MiniMaxH3Ref2VAModularPipeline,
)


def _write_partial_checkpoint(model_dir: Path, component_dirs: tuple[str, ...]) -> None:
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text(
        json.dumps({
            "_class_name": "MiniMaxH3ModularPipeline",
            "_diffusers_version": "0.35.0",
        }))
    for component_dir in component_dirs:
        (model_dir / component_dir).mkdir()


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
    shared_dirs = {
        "audio_scheduler",
        "audio_vae",
        "processor",
        "scheduler",
        "text_encoder",
        "tokenizer",
        "vae",
    }

    assert standard_dirs == shared_dirs | {"transformer"}
    assert ref2va_dirs == shared_dirs | {"transformer_ref"}


@pytest.mark.parametrize(
    ("pipeline_cls", "override_pipeline_cls_name", "included_transformer", "excluded_transformer"),
    [
        (MiniMaxH3ModularPipeline, None, "transformer/**", "transformer_ref/**"),
        (
            MiniMaxH3Ref2VAModularPipeline,
            "MiniMaxH3Ref2VAModularPipeline",
            "transformer_ref/**",
            "transformer/**",
        ),
    ],
)
def test_build_pipeline_forwards_real_minimax_patterns(
    pipeline_cls,
    override_pipeline_cls_name: str | None,
    included_transformer: str,
    excluded_transformer: str,
    monkeypatch,
) -> None:
    captured: dict = {}

    def fake_get_model_info(**kwargs):
        captured["override_pipeline_cls_name"] = kwargs["override_pipeline_cls_name"]
        return SimpleNamespace(pipeline_cls=pipeline_cls)

    monkeypatch.setattr(
        pipelines,
        "get_model_info",
        fake_get_model_info,
    )
    monkeypatch.setattr(
        pipeline_cls,
        "__init__",
        lambda self, model_path, fastvideo_args: captured.update(pipeline_model_path=model_path),
    )

    def fake_download(model_path, **kwargs):
        captured.update(kwargs)
        return "/tmp/minimax-h3"

    monkeypatch.setattr(pipelines, "maybe_download_model", fake_download)
    args = SimpleNamespace(
        model_path="MiniMaxAI/MiniMax-H3",
        revision="test-revision",
        workload_type=None,
        override_pipeline_cls_name=override_pipeline_cls_name,
    )

    pipelines.build_pipeline(args)

    assert included_transformer in captured["allow_patterns"]
    assert excluded_transformer not in captured["allow_patterns"]
    assert "FL2VA/**" not in captured["allow_patterns"]
    assert "Ref2VA/**" not in captured["allow_patterns"]
    assert captured["override_pipeline_cls_name"] == override_pipeline_cls_name
    assert captured["revision"] == "test-revision"
    assert captured["pipeline_model_path"] == "/tmp/minimax-h3"


@pytest.mark.parametrize(
    ("pipeline_cls", "selected_transformer", "excluded_transformer"),
    [
        (MiniMaxH3ModularPipeline, "transformer/model.safetensors", "transformer_ref/model.safetensors"),
        (MiniMaxH3Ref2VAModularPipeline, "transformer_ref/model.safetensors", "transformer/model.safetensors"),
    ],
)
def test_minimax_patterns_filter_unused_repo_partitions(
    pipeline_cls,
    selected_transformer: str,
    excluded_transformer: str,
) -> None:
    repo_files = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "transformer/model.safetensors",
        "transformer_ref/model.safetensors",
        "FL2VA/model.safetensors",
        "Ref2VA/model.safetensors",
    ]

    selected = set(filter_repo_objects(
        repo_files,
        allow_patterns=pipeline_cls.get_hf_download_allow_patterns(),
    ))

    assert "model_index.json" in selected
    assert "scheduler/scheduler_config.json" in selected
    assert selected_transformer in selected
    assert excluded_transformer not in selected
    assert "FL2VA/model.safetensors" not in selected
    assert "Ref2VA/model.safetensors" not in selected


def test_direct_pipeline_load_forwards_partial_patterns(tmp_path: Path, monkeypatch) -> None:
    component_dirs = MiniMaxH3Ref2VAModularPipeline.get_hf_download_component_dirs()
    model_dir = tmp_path / "minimax-h3-ref2va"
    _write_partial_checkpoint(model_dir, component_dirs)
    captured: dict = {}

    def fake_download(model_path, **kwargs):
        captured.update(kwargs)
        return str(model_dir)

    monkeypatch.setattr("fastvideo.pipelines.composed_pipeline_base.maybe_download_model", fake_download)
    pipeline = object.__new__(MiniMaxH3Ref2VAModularPipeline)
    pipeline.model_path = "MiniMaxAI/MiniMax-H3"
    pipeline.fastvideo_args = SimpleNamespace(revision="test-revision")

    pipeline._load_config(pipeline.model_path)

    assert "transformer_ref/**" in captured["allow_patterns"]
    assert "transformer/**" not in captured["allow_patterns"]
    assert captured["revision"] == "test-revision"


def test_umbrella_download_prefixes_selected_patterns(tmp_path: Path, monkeypatch) -> None:
    captured: dict = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        (tmp_path / "minimax-h3").mkdir()
        return str(tmp_path)

    monkeypatch.setattr(utils, "snapshot_download", fake_snapshot_download)

    model_path = utils.maybe_download_model(
        "org/repo/minimax-h3",
        local_dir=str(tmp_path),
        allow_patterns=["model_index.json", "transformer_ref/**"],
    )

    assert captured["repo_id"] == "org/repo"
    assert captured["allow_patterns"] == [
        "minimax-h3/model_index.json",
        "minimax-h3/transformer_ref/**",
    ]
    assert model_path == str(tmp_path / "minimax-h3")


def test_incomplete_local_checkpoint_fails_selected_component_validation(tmp_path: Path, monkeypatch) -> None:
    model_dir = tmp_path / "minimax-h3"
    ref2va_dirs = MiniMaxH3Ref2VAModularPipeline.get_hf_download_component_dirs()
    standard_dirs = tuple("transformer" if component_dir == "transformer_ref" else component_dir
                          for component_dir in ref2va_dirs)
    _write_partial_checkpoint(model_dir, standard_dirs)
    monkeypatch.setattr(utils, "snapshot_download", lambda **kwargs: pytest.fail("local paths must not hit the Hub"))

    resolved_path = utils.maybe_download_model(
        str(model_dir),
        allow_patterns=MiniMaxH3Ref2VAModularPipeline.get_hf_download_allow_patterns(),
    )

    with pytest.raises(ValueError, match="transformer_ref"):
        utils.verify_model_config_and_directory(
            resolved_path,
            required_component_dirs=ref2va_dirs,
        )


def test_model_index_supports_umbrella_repo_paths(tmp_path: Path, monkeypatch) -> None:
    captured: dict = {}

    def fake_hf_hub_download(**kwargs):
        captured.update(kwargs)
        manifest_path = tmp_path / kwargs["filename"]
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps({
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
