# SPDX-License-Identifier: Apache-2.0
"""Path-routing tests for fastvideo.utils.maybe_download_lora."""
from __future__ import annotations

import huggingface_hub

import fastvideo.utils as fv_utils
from fastvideo.utils import maybe_download_lora


def test_triple_slash_downloads_single_file(monkeypatch):
    """org/repo/sub/file.safetensors -> hf_hub_download(repo_id=org/repo, filename=sub/file)."""
    calls: list[dict] = []

    def fake_hf_hub_download(*, repo_id, filename, local_dir=None, **kwargs):
        calls.append({"repo_id": repo_id, "filename": filename, "local_dir": local_dir})
        return f"/cache/{repo_id}/{filename}"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)

    result = maybe_download_lora("vita-video-gen/svi-model/version-1.0/svi-shot.safetensors")

    assert len(calls) == 1
    assert calls[0]["repo_id"] == "vita-video-gen/svi-model"
    assert calls[0]["filename"] == "version-1.0/svi-shot.safetensors"
    assert result == "/cache/vita-video-gen/svi-model/version-1.0/svi-shot.safetensors"


def test_triple_slash_keeps_only_first_two_segments_as_repo(monkeypatch):
    """A deeper nesting still maps repo_id to the first two segments."""
    captured: dict = {}

    def fake_hf_hub_download(*, repo_id, filename, local_dir=None, **kwargs):
        captured["repo_id"] = repo_id
        captured["filename"] = filename
        return "ok"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)

    maybe_download_lora("org/repo/a/b/c/weights.safetensors")

    assert captured["repo_id"] == "org/repo"
    assert captured["filename"] == "a/b/c/weights.safetensors"


def test_local_file_short_circuits(tmp_path, monkeypatch):
    """An existing local .safetensors file is returned verbatim, no download."""

    def boom(*args, **kwargs):
        raise AssertionError("hf_hub_download must not be called for a local file")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)

    f = tmp_path / "version-1.0" / "svi-shot.safetensors"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"\x00")

    assert maybe_download_lora(str(f)) == str(f)


def test_plain_repo_id_falls_through_to_repo_download(monkeypatch):
    """A two-segment HF id (no .safetensors suffix) does NOT hit the single-file branch."""

    def boom(*args, **kwargs):
        raise AssertionError("two-segment repo id must not take the single-file branch")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    monkeypatch.setattr(fv_utils, "maybe_download_model", lambda *a, **k: "/cache/org/repo")
    monkeypatch.setattr(fv_utils, "_best_guess_weight_name", lambda *a, **k: "adapter.safetensors")

    result = maybe_download_lora("org/repo")

    assert result == "/cache/org/repo/adapter.safetensors"
