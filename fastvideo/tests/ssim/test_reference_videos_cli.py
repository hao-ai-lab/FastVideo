# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from fastvideo.tests.ssim import reference_videos_cli


class _FakeHfApi:
    instances = []
    repo_files = []

    def __init__(self, token):
        self.token = token
        self.created_repos = []
        self.uploaded_files = []
        self.uploaded_folders = []
        _FakeHfApi.instances.append(self)

    def create_repo(self, **kwargs):
        self.created_repos.append(kwargs)

    def upload_file(self, **kwargs):
        self.uploaded_files.append(kwargs)

    def upload_folder(self, **kwargs):
        self.uploaded_folders.append(kwargs)

    def list_repo_files(self, **kwargs):
        return list(self.repo_files)


def _fake_snapshot_download(*, allow_patterns, cache_dir, **_kwargs):
    assert len(allow_patterns) == 1
    draft_prefix = allow_patterns[0].removesuffix("/**")
    snapshot_root = Path(cache_dir) / "snapshot"
    draft_root = snapshot_root / draft_prefix
    draft_root.mkdir(parents=True)
    (draft_root / "sample.mp4").write_text("draft", encoding="utf-8")
    return str(snapshot_root)


@pytest.fixture(autouse=True)
def _fake_hf(monkeypatch):
    _FakeHfApi.instances = []
    _FakeHfApi.repo_files = []
    monkeypatch.setattr(
        reference_videos_cli,
        "_load_hf_sdk",
        lambda: (_FakeHfApi, _fake_snapshot_download),
    )


def test_upload_draft_reference_artifact_uses_drafts_prefix(tmp_path):
    ssim_dir = tmp_path / "ssim"
    generated = (
        ssim_dir
        / "generated_videos"
        / "default"
        / "L40S_reference_videos"
        / "model-a"
        / "flash"
        / "sample.mp4"
    )
    generated.parent.mkdir(parents=True)
    generated.write_text("video", encoding="utf-8")
    reference_folder = (
        ssim_dir
        / "reference_videos"
        / "default"
        / "L40S_reference_videos"
        / "model-a"
        / "flash"
    )

    draft_path = reference_videos_cli.upload_draft_reference_artifact(
        repo_id="FastVideo/ssim-reference-videos",
        repo_type="dataset",
        generated_artifact_path=generated,
        reference_folder=reference_folder,
        token="hf_token",
        base_dir=ssim_dir,
    )

    assert draft_path == "drafts/default/L40S_reference_videos/model-a/flash/sample.mp4"
    fake_api = _FakeHfApi.instances[-1]
    assert fake_api.uploaded_files[0]["path_in_repo"] == draft_path


def test_promote_draft_refuses_existing_canonical_reference():
    _FakeHfApi.repo_files = [
        "drafts/default/L40S_reference_videos/model-a/flash/sample.mp4",
        "reference_videos/default/L40S_reference_videos/model-a/flash/sample.mp4",
    ]

    with pytest.raises(RuntimeError, match="Refusing to overwrite existing HF files"):
        reference_videos_cli.promote_draft_references(
            repo_id="FastVideo/ssim-reference-videos",
            repo_type="dataset",
            quality_tier="default",
            device_folder="L40S_reference_videos",
            model_id="model-a",
            attention_backend="flash",
            token="hf_token",
        )


def test_promote_draft_uploads_downloaded_draft_folder(tmp_path, monkeypatch):
    _FakeHfApi.repo_files = [
        "drafts/default/L40S_reference_videos/model-a/flash/sample.mp4",
    ]

    def fake_snapshot_download(*, allow_patterns, cache_dir, **_kwargs):
        snapshot_root = tmp_path / "snapshot"
        draft_root = snapshot_root / allow_patterns[0].removesuffix("/**")
        draft_root.mkdir(parents=True)
        (draft_root / "sample.mp4").write_text("draft", encoding="utf-8")
        return str(snapshot_root)

    monkeypatch.setattr(
        reference_videos_cli,
        "_load_hf_sdk",
        lambda: (_FakeHfApi, fake_snapshot_download),
    )

    reference_videos_cli.promote_draft_references(
        repo_id="FastVideo/ssim-reference-videos",
        repo_type="dataset",
        quality_tier="default",
        device_folder="L40S_reference_videos",
        model_id="model-a",
        attention_backend="flash",
        token="hf_token",
    )

    fake_api = _FakeHfApi.instances[-1]
    upload = fake_api.uploaded_folders[0]
    assert upload["path_in_repo"] == "reference_videos/default/L40S_reference_videos/model-a/flash"
    assert Path(upload["folder_path"]).name == "flash"
    shutil.rmtree(tmp_path / "snapshot", ignore_errors=True)
