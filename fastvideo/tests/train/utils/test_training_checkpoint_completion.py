# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for strict training-checkpoint publication and retention.

Only temporary files and fake DCP/RNG writers are used. Real distributed save
coordination and CUDA RNG serialization remain integration-test concerns.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fastvideo.train.utils import checkpoint as checkpoints


def _raw_config(world_size: int = 2) -> dict:
    return {"training": {"distributed": {"num_gpus": world_size}}}


def _published(output: Path, step: int, *, world_size: int = 2) -> Path:
    directory = output / f"checkpoint-{step}"
    (directory / "dcp").mkdir(parents=True)
    (directory / "dcp" / ".metadata").write_bytes(b"fake-dcp-metadata")
    (directory / "metadata.json").write_text(
        json.dumps({"step": step, "config": _raw_config(world_size)}), encoding="utf-8")
    for rank in range(world_size):
        (directory / f"rng_state_rank{rank}.pt").write_bytes(b"fake-rng-state")
    (directory / ".complete").write_text("complete\n", encoding="utf-8")
    return directory


def _manager(output: Path, *, keep_last: int = 0) -> checkpoints.CheckpointManager:
    return checkpoints.CheckpointManager(
        method=SimpleNamespace(checkpoint_state=lambda: {}),
        dataloader=None,
        output_dir=str(output),
        config=checkpoints.CheckpointConfig(
            save_steps=1, keep_last=keep_last, require_complete_training_checkpoint=True),
        raw_config=_raw_config(),
    )


@pytest.fixture(autouse=True)
def _rank_zero(monkeypatch):
    monkeypatch.setattr(checkpoints, "_rank", lambda: 0)
    monkeypatch.setattr(checkpoints, "_barrier", lambda: None)


def test_strict_latest_skips_unpublished_newer_save_but_legacy_mode_accepts_it(tmp_path):
    older = _published(tmp_path, 5)
    newer = _published(tmp_path, 10)
    (newer / ".complete").unlink()

    assert checkpoints._find_latest_checkpoint(tmp_path, require_complete_marker=True) == older
    assert checkpoints._find_latest_checkpoint(tmp_path) == newer
    assert checkpoints._resolve_resume_checkpoint(
        "latest", output_dir=str(tmp_path), require_complete_marker=True) == older
    for path in (newer, newer / "dcp"):
        with pytest.raises(ValueError, match="incomplete"):
            checkpoints._resolve_resume_checkpoint(str(path), output_dir=str(tmp_path), require_complete_marker=True)


@pytest.mark.parametrize("defect", [
    "missing_dcp_metadata", "invalid_marker", "wrong_step", "invalid_metadata", "missing_rng", "empty_rng", "extra_rng",
    "invalid_world_size",
])
def test_strict_resume_rejects_partial_or_inconsistent_publication(tmp_path, defect):
    directory = _published(tmp_path, 10)
    if defect == "missing_dcp_metadata":
        (directory / "dcp" / ".metadata").unlink()
    elif defect == "invalid_marker":
        (directory / ".complete").write_text("incomplete\n", encoding="utf-8")
    elif defect == "wrong_step":
        (directory / "metadata.json").write_text(json.dumps({"step": 9, "config": _raw_config()}), encoding="utf-8")
    elif defect == "invalid_metadata":
        (directory / "metadata.json").write_text("{truncated", encoding="utf-8")
    elif defect == "missing_rng":
        (directory / "rng_state_rank1.pt").unlink()
    elif defect == "empty_rng":
        (directory / "rng_state_rank1.pt").write_bytes(b"")
    elif defect == "extra_rng":
        (directory / "rng_state_rank2.pt").write_bytes(b"stale")
    elif defect == "invalid_world_size":
        (directory / "metadata.json").write_text(json.dumps({"step": 10, "config": _raw_config(True)}), encoding="utf-8")

    assert not checkpoints._is_complete_training_checkpoint(directory, require_complete_marker=True)
    # Existing incomplete state must not silently become a fresh training run.
    with pytest.raises(ValueError, match="refusing to start from scratch"):
        checkpoints._resolve_resume_checkpoint("latest", output_dir=str(tmp_path), require_complete_marker=True)


def test_strict_latest_allows_a_fresh_empty_namespace(tmp_path):
    assert checkpoints._resolve_resume_checkpoint(
        "latest", output_dir=str(tmp_path), require_complete_marker=True) is None


def test_strict_manager_requires_saved_world_size(tmp_path):
    with pytest.raises(ValueError, match="num_gpus"):
        checkpoints.CheckpointManager(
            method=None, dataloader=None, output_dir=str(tmp_path), raw_config={},
            config=checkpoints.CheckpointConfig(save_steps=1, keep_last=1, require_complete_training_checkpoint=True))


def test_save_removes_old_marker_and_publishes_after_all_rank_rng_barrier(tmp_path, monkeypatch):
    directory = _published(tmp_path, 10)
    marker = directory / ".complete"
    # Simulate a retry with an old marker. No RNG from this retry exists yet.
    for path in directory.glob("rng_state_rank*.pt"):
        path.unlink()
    manager = _manager(tmp_path)
    events = []

    def save_dcp(states, *, checkpoint_id):
        assert states == {} and Path(checkpoint_id) == directory / "dcp"
        assert not marker.exists()
        events.append("dcp")
        (Path(checkpoint_id) / ".metadata").write_bytes(b"new-dcp-metadata")

    def save_rng(path):
        assert path == directory and not marker.exists()
        events.append("rng")
        (path / "rng_state_rank0.pt").write_bytes(b"new-rank-zero-rng")

    def barrier():
        if events and events[-1] == "rng":
            # Simulate the peer finishing its RNG write before this collective
            # returns. The complete marker must still be absent at this point.
            assert not marker.exists()
            (directory / "rng_state_rank1.pt").write_bytes(b"new-peer-rng")
        events.append("barrier")

    publish = checkpoints._publish_training_checkpoint_complete

    def publish_after_barrier(path):
        assert events[-2:] == ["rng", "barrier"]
        assert (path / "rng_state_rank1.pt").is_file()
        events.append("publish")
        publish(path)

    monkeypatch.setattr(checkpoints.dcp, "save", save_dcp)
    monkeypatch.setattr(manager, "_save_rng_snapshot", save_rng)
    monkeypatch.setattr(checkpoints, "_barrier", barrier)
    monkeypatch.setattr(checkpoints, "_publish_training_checkpoint_complete", publish_after_barrier)
    manager.save(10)

    assert events == ["barrier", "dcp", "barrier", "rng", "barrier", "publish", "barrier"]
    assert checkpoints._is_complete_training_checkpoint(directory, require_complete_marker=True)
    assert manager._last_saved_step == 10
    assert not list(directory.glob(".complete.tmp-*"))


@pytest.mark.parametrize("failure_stage", ["dcp", "rng"])
def test_failed_retry_is_not_resumable_and_does_not_prune_last_good_save(tmp_path, monkeypatch, failure_stage):
    older = _published(tmp_path, 5)
    retry = _published(tmp_path, 10)
    manager = _manager(tmp_path, keep_last=1)

    def save_dcp(_states, *, checkpoint_id):
        assert not (retry / ".complete").exists()
        if failure_stage == "dcp":
            raise RuntimeError("synthetic DCP failure")
        (Path(checkpoint_id) / ".metadata").write_bytes(b"new-dcp-metadata")

    def save_rng(_directory):
        raise RuntimeError("synthetic RNG failure")

    monkeypatch.setattr(checkpoints.dcp, "save", save_dcp)
    monkeypatch.setattr(manager, "_save_rng_snapshot", save_rng)
    with pytest.raises(RuntimeError, match="synthetic"):
        manager.save(10)

    assert not (retry / ".complete").exists()
    assert manager._last_saved_step is None
    assert checkpoints._find_latest_checkpoint(tmp_path, require_complete_marker=True) == older
    assert older.is_dir() and retry.is_dir()


def test_strict_retention_counts_only_complete_training_saves(tmp_path):
    directories = {step: _published(tmp_path, step) for step in (100, 200, 300, 400)}
    (directories[400] / ".complete").unlink()
    inference = tmp_path / "inference" / "checkpoint-50"
    inference.mkdir(parents=True)
    (inference / ".complete").write_text("complete\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("keep", encoding="utf-8")

    _manager(tmp_path, keep_last=2)._cleanup_old_checkpoints()

    assert not directories[100].exists()
    assert all(directories[step].is_dir() for step in (200, 300, 400))
    assert inference.is_dir()
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "keep"
