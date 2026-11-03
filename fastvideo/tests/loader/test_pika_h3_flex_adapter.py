# SPDX-License-Identifier: Apache-2.0
"""Pika H3 Flex adapter discovery and manifest contract."""

import json
from pathlib import Path

import pytest

from fastvideo.pipelines.basic.pika_h3_flex import adapter


def _manifest(task: str = "fl2va") -> dict:
    return {
        "format_version": 1,
        "model": "pika-flex",
        "task": task,
        "release_id": "pika-flex-fl2va-v1",
        "rank": 128,
        "alpha": 128,
        "trunk": "transformer_ref" if task == "ref2va" else "transformer",
        "sampling": {
            "knots": [1000, 750, 500, 250, 100, 40, 0],
            "video_shift": 12.0,
            "audio_shift": 3.0,
            "method": "renoise",
        },
        "base": {"repo_id": "MiniMaxAI/MiniMax-H3", "revision": "42ed227ee7df40d41602854ae760620d6eb651fe"},
        "target_modules": ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
        "sha256": "0" * 64,
    }


def _write_release(root: Path, task: str = "fl2va") -> Path:
    task_dir = root / task
    task_dir.mkdir(parents=True)
    (task_dir / "adapter.safetensors").write_bytes(b"")
    (task_dir / "manifest.json").write_text(json.dumps(_manifest(task)), encoding="utf-8")
    return task_dir


def test_manifest_matches_released_contract() -> None:
    adapter.validate_adapter_manifest(_manifest(), task="fl2va")
    adapter.validate_adapter_manifest(_manifest("ref2va"), task="ref2va")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rank", 64, "rank"),
        ("alpha", 64, "alpha"),
        ("trunk", "transformer_ref", "trunk"),
        ("task", "ref2va", "task"),
        ("format_version", 2, "format"),
    ],
)
def test_manifest_rejects_wrong_release(field: str, value: object, message: str) -> None:
    manifest = _manifest()
    manifest[field] = value
    with pytest.raises(adapter.PikaAdapterError, match=message):
        adapter.validate_adapter_manifest(manifest, task="fl2va")


def test_manifest_rejects_a_different_sampler() -> None:
    manifest = _manifest()
    manifest["sampling"]["knots"] = [1000, 500, 0]
    with pytest.raises(adapter.PikaAdapterError, match="sampling knots"):
        adapter.validate_adapter_manifest(manifest, task="fl2va")


def test_load_adapter_finds_the_task_directory(tmp_path: Path) -> None:
    task_dir = _write_release(tmp_path)
    loaded = adapter.load_adapter(tmp_path, task="fl2va")
    assert loaded.path == str(task_dir / "adapter.safetensors")
    assert loaded.trunk == "transformer"
    assert loaded.manifest["task"] == "fl2va"


def test_load_adapter_finds_a_task_directory_or_adapter_file(tmp_path: Path) -> None:
    task_dir = _write_release(tmp_path)
    assert adapter.load_adapter(task_dir, task="fl2va").path == str(task_dir / "adapter.safetensors")
    assert adapter.load_adapter(task_dir / "adapter.safetensors", task="fl2va").path == str(
        task_dir / "adapter.safetensors")


def test_load_adapter_rejects_a_missing_task(tmp_path: Path) -> None:
    _write_release(tmp_path, "fl2va")
    with pytest.raises(adapter.PikaAdapterError, match="ref2va"):
        adapter.load_adapter(tmp_path, task="ref2va")


def test_load_adapter_requires_the_manifest(tmp_path: Path) -> None:
    task_dir = _write_release(tmp_path)
    (task_dir / "manifest.json").unlink()
    with pytest.raises(adapter.PikaAdapterError, match="manifest"):
        adapter.load_adapter(tmp_path, task="fl2va")
