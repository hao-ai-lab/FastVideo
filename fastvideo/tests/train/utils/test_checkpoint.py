# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for :mod:`fastvideo.train.utils.checkpoint`.

Covers the pure-Python portions of the checkpoint manager: name
parsing, resume-path resolution, metadata round-trip, rolling-delete
cleanup, the ``_is_stateful`` predicate, and the ``maybe_save`` gating
logic. Code paths that touch DCP (``dcp.save`` / ``dcp.load``) and
CUDA RNG snapshots are intentionally not covered here — those need a
GPU runner and will be tested in later phases.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from fastvideo.train.utils import checkpoint as checkpoint_utils
from fastvideo.train.utils.checkpoint import (
    CheckpointConfig,
    CheckpointManager,
    _find_latest_checkpoint,
    _is_stateful,
    _parse_step_from_dir,
    _resolve_resume_checkpoint,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_checkpoint_dir(
    output_dir: Path,
    step: int,
    *,
    with_dcp: bool = True,
    complete: bool = True,
) -> Path:
    """Create a fake ``checkpoint-<step>/dcp`` directory tree."""
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if with_dcp:
        (ckpt_dir / "dcp").mkdir(exist_ok=True)
    if complete:
        (ckpt_dir / ".complete").write_text("complete\n")
    return ckpt_dir


def _make_manager(
    tmp_path: Path,
    *,
    save_steps: int = 0,
    keep_last: int = 0,
    raw_config: dict[str, Any] | None = None,
) -> CheckpointManager:
    """Build a minimal ``CheckpointManager`` for tests that don't touch DCP."""
    return CheckpointManager(
        method=None,
        dataloader=None,
        output_dir=str(tmp_path),
        config=CheckpointConfig(save_steps=save_steps, keep_last=keep_last),
        raw_config=raw_config,
    )


# ---------------------------------------------------------------------------
# A. _is_stateful predicate
# ---------------------------------------------------------------------------


class _Full:

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        pass


class _MissingStateDict:

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        pass


class _MissingLoad:

    def state_dict(self) -> dict[str, Any]:
        return {}


def test_is_stateful_true_for_full_object() -> None:
    assert _is_stateful(_Full()) is True


def test_is_stateful_false_when_missing_state_dict() -> None:
    assert _is_stateful(_MissingStateDict()) is False


def test_is_stateful_false_when_missing_load_state_dict() -> None:
    assert _is_stateful(_MissingLoad()) is False


# ---------------------------------------------------------------------------
# B. _parse_step_from_dir
# ---------------------------------------------------------------------------


def test_parse_step_valid(tmp_path: Path) -> None:
    assert _parse_step_from_dir(tmp_path / "checkpoint-100") == 100


def test_parse_step_zero(tmp_path: Path) -> None:
    assert _parse_step_from_dir(tmp_path / "checkpoint-0") == 0


def test_parse_step_invalid_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid checkpoint directory"):
        _parse_step_from_dir(tmp_path / "not-a-checkpoint")


# ---------------------------------------------------------------------------
# C. _find_latest_checkpoint
# ---------------------------------------------------------------------------


def test_find_latest_returns_none_on_nonexistent_dir(tmp_path: Path) -> None:
    assert _find_latest_checkpoint(tmp_path / "missing") is None


def test_find_latest_returns_none_on_empty_dir(tmp_path: Path) -> None:
    assert _find_latest_checkpoint(tmp_path) is None


def test_find_latest_returns_largest_step(tmp_path: Path) -> None:
    _make_checkpoint_dir(tmp_path, 10)
    _make_checkpoint_dir(tmp_path, 200)
    _make_checkpoint_dir(tmp_path, 50)
    latest = _find_latest_checkpoint(tmp_path)
    assert latest is not None
    assert latest.name == "checkpoint-200"


def test_find_latest_skips_dirs_without_dcp_subdir(tmp_path: Path) -> None:
    # checkpoint-10 is "corrupted" — has no dcp/ subdir, must be skipped.
    _make_checkpoint_dir(tmp_path, 10, with_dcp=False)
    _make_checkpoint_dir(tmp_path, 5, with_dcp=True)
    latest = _find_latest_checkpoint(tmp_path)
    assert latest is not None
    assert latest.name == "checkpoint-5"


def test_find_latest_falls_back_from_incomplete_new_checkpoint(
        tmp_path: Path) -> None:
    _make_checkpoint_dir(tmp_path, 100)
    incomplete = _make_checkpoint_dir(tmp_path, 200, complete=False)
    (incomplete / "dcp" / ".metadata").write_text("dcp metadata")
    (incomplete / "metadata.json").write_text(
        '{"step": 200, "completion_marker": ".complete"}')

    latest = _find_latest_checkpoint(tmp_path)

    assert latest is not None
    assert latest.name == "checkpoint-100"


def test_find_latest_accepts_legacy_dcp_metadata(tmp_path: Path) -> None:
    legacy = _make_checkpoint_dir(tmp_path, 75, complete=False)
    (legacy / "dcp" / ".metadata").write_text("dcp metadata")

    latest = _find_latest_checkpoint(tmp_path)

    assert latest == legacy


def test_resolve_explicit_incomplete_new_checkpoint_raises(
        tmp_path: Path) -> None:
    incomplete = _make_checkpoint_dir(tmp_path, 200, complete=False)
    (incomplete / "dcp" / ".metadata").write_text("dcp metadata")
    (incomplete / "metadata.json").write_text(
        '{"step": 200, "completion_marker": ".complete"}')

    with pytest.raises(FileNotFoundError, match="did not complete"):
        _resolve_resume_checkpoint(str(incomplete), output_dir=str(tmp_path))


def test_find_latest_skips_non_checkpoint_dirs(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()
    (tmp_path / "wandb").mkdir()
    (tmp_path / "some_file.txt").write_text("noise")
    _make_checkpoint_dir(tmp_path, 7)
    latest = _find_latest_checkpoint(tmp_path)
    assert latest is not None
    assert latest.name == "checkpoint-7"


# ---------------------------------------------------------------------------
# D. _resolve_resume_checkpoint
# ---------------------------------------------------------------------------


def test_resolve_latest_with_no_checkpoints_returns_none(tmp_path: Path) -> None:
    out = tmp_path / "outputs"
    out.mkdir()
    assert _resolve_resume_checkpoint("latest", output_dir=str(out)) is None


def test_resolve_latest_returns_latest_checkpoint(tmp_path: Path) -> None:
    _make_checkpoint_dir(tmp_path, 30)
    _make_checkpoint_dir(tmp_path, 10)
    resolved = _resolve_resume_checkpoint("latest", output_dir=str(tmp_path))
    assert resolved is not None
    assert resolved.name == "checkpoint-30"


def test_resolve_explicit_checkpoint_dir(tmp_path: Path) -> None:
    ckpt = _make_checkpoint_dir(tmp_path, 42)
    resolved = _resolve_resume_checkpoint(str(ckpt), output_dir=str(tmp_path))
    assert resolved is not None
    assert resolved.name == "checkpoint-42"


def test_resolve_dcp_subdir_returns_parent_checkpoint(tmp_path: Path) -> None:
    ckpt = _make_checkpoint_dir(tmp_path, 42)
    dcp_path = ckpt / "dcp"
    resolved = _resolve_resume_checkpoint(str(dcp_path), output_dir=str(tmp_path))
    assert resolved is not None
    assert resolved.name == "checkpoint-42"


def test_resolve_output_dir_returns_latest(tmp_path: Path) -> None:
    out = tmp_path / "outputs"
    out.mkdir()
    _make_checkpoint_dir(out, 100)
    _make_checkpoint_dir(out, 50)
    resolved = _resolve_resume_checkpoint(str(out), output_dir=str(tmp_path))
    assert resolved is not None
    assert resolved.name == "checkpoint-100"


def test_resolve_nonexistent_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _resolve_resume_checkpoint(str(tmp_path / "missing"), output_dir=str(tmp_path))


def test_resolve_checkpoint_without_dcp_raises(tmp_path: Path) -> None:
    ckpt = _make_checkpoint_dir(tmp_path, 5, with_dcp=False)
    with pytest.raises(FileNotFoundError, match="dcp"):
        _resolve_resume_checkpoint(str(ckpt), output_dir=str(tmp_path))


def test_resolve_unknown_dir_raises(tmp_path: Path) -> None:
    """A dir that is neither a checkpoint nor an output_dir-with-checkpoints."""
    bogus = tmp_path / "bogus"
    bogus.mkdir()
    with pytest.raises(ValueError, match="Could not resolve"):
        _resolve_resume_checkpoint(str(bogus), output_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# E. metadata read/write
# ---------------------------------------------------------------------------


def test_write_metadata_roundtrip_with_step(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    ckpt_dir = _make_checkpoint_dir(tmp_path, 7)
    mgr._write_metadata(ckpt_dir, step=7)
    loaded = CheckpointManager.load_metadata(ckpt_dir)
    assert loaded == {"step": 7, "completion_marker": ".complete"}


def test_write_metadata_includes_raw_config(tmp_path: Path) -> None:
    raw = {
        "models": {
            "student": {
                "_target_": "X"
            }
        },
        "training": {
            "distributed": {
                "num_gpus": 4
            }
        },
    }
    mgr = _make_manager(tmp_path, raw_config=raw)
    ckpt_dir = _make_checkpoint_dir(tmp_path, 7)
    mgr._write_metadata(ckpt_dir, step=7)
    loaded = CheckpointManager.load_metadata(ckpt_dir)
    assert loaded["step"] == 7
    assert loaded["config"] == raw


def test_load_metadata_raises_on_missing_file(tmp_path: Path) -> None:
    ckpt_dir = _make_checkpoint_dir(tmp_path, 7)
    # No metadata.json written.
    with pytest.raises(FileNotFoundError, match="metadata"):
        CheckpointManager.load_metadata(ckpt_dir)


def test_save_marks_complete_after_dcp_and_rng_barriers(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _Method:

        @staticmethod
        def checkpoint_state() -> dict[str, Any]:
            return {}

    manager = CheckpointManager(
        method=_Method(),
        dataloader=None,
        output_dir=str(tmp_path),
        config=CheckpointConfig(save_steps=100, keep_last=0),
    )
    events = []
    monkeypatch.setattr(checkpoint_utils, "_barrier", lambda: events.append("barrier"))
    monkeypatch.setattr(checkpoint_utils.dcp, "save", lambda *args, **kwargs: events.append("dcp"))
    monkeypatch.setattr(manager, "_save_rng_snapshot", lambda checkpoint: events.append("rng"))
    monkeypatch.setattr(manager, "_cleanup_old_checkpoints", lambda: events.append("cleanup"))
    original_replace = checkpoint_utils.os.replace

    def record_replace(source: Path, destination: Path) -> None:
        events.append("marker")
        original_replace(source, destination)

    monkeypatch.setattr(checkpoint_utils.os, "replace", record_replace)

    manager.save(100)

    assert events == [
        "barrier",
        "dcp",
        "barrier",
        "rng",
        "barrier",
        "marker",
        "barrier",
        "cleanup",
    ]
    assert (tmp_path / "checkpoint-100" / ".complete").is_file()


# ---------------------------------------------------------------------------
# F. _cleanup_old_checkpoints (rolling delete)
# ---------------------------------------------------------------------------


def test_cleanup_keep_last_zero_is_noop(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path, keep_last=0)
    for step in (1, 2, 3):
        _make_checkpoint_dir(tmp_path, step)
    mgr._cleanup_old_checkpoints()
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["checkpoint-1", "checkpoint-2", "checkpoint-3"]


def test_cleanup_keeps_newest_when_over_limit(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path, keep_last=2)
    for step in (1, 5, 10, 50, 100):
        _make_checkpoint_dir(tmp_path, step)
    mgr._cleanup_old_checkpoints()
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["checkpoint-100", "checkpoint-50"]


def test_cleanup_no_op_when_under_limit(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path, keep_last=3)
    for step in (1, 2):
        _make_checkpoint_dir(tmp_path, step)
    mgr._cleanup_old_checkpoints()
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["checkpoint-1", "checkpoint-2"]


def test_cleanup_skips_non_checkpoint_dirs(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path, keep_last=1)
    for step in (1, 2, 3):
        _make_checkpoint_dir(tmp_path, step)
    (tmp_path / "logs").mkdir()
    (tmp_path / "wandb").mkdir()
    mgr._cleanup_old_checkpoints()
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["checkpoint-3", "logs", "wandb"]


# ---------------------------------------------------------------------------
# G. maybe_save gating logic
# ---------------------------------------------------------------------------


def _record_save_calls(mgr: CheckpointManager) -> list[int]:
    """Replace ``mgr.save`` with a recorder that mimics the side effect
    of advancing ``_last_saved_step`` so dedup logic still works."""
    calls: list[int] = []

    def fake_save(step: int) -> None:
        calls.append(step)
        mgr._last_saved_step = step

    mgr.save = fake_save  # type: ignore[method-assign]
    return calls


def test_maybe_save_skipped_when_save_steps_is_zero(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path, save_steps=0)
    calls = _record_save_calls(mgr)
    mgr.maybe_save(step=10)
    mgr.maybe_save(step=100)
    assert calls == []


def test_maybe_save_skipped_when_step_not_on_interval(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path, save_steps=10)
    calls = _record_save_calls(mgr)
    for step in (1, 5, 9, 11, 15):
        mgr.maybe_save(step=step)
    assert calls == []


def test_maybe_save_dedupes_on_repeated_call(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path, save_steps=10)
    calls = _record_save_calls(mgr)
    mgr.maybe_save(step=20)
    mgr.maybe_save(step=20)
    mgr.maybe_save(step=20)
    assert calls == [20]


def test_maybe_save_triggers_on_each_interval(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path, save_steps=10)
    calls = _record_save_calls(mgr)
    for step in range(1, 41):
        mgr.maybe_save(step=step)
    assert calls == [10, 20, 30, 40]


def test_save_final_dedupes_step_saved_on_interval(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path, save_steps=100)
    calls = _record_save_calls(mgr)

    mgr.maybe_save(step=500)
    mgr.save_final(step=500)

    assert calls == [500]


def test_save_final_dedupes_terminal_resumed_step(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_checkpoint_dir(tmp_path, 500)
    mgr = _make_manager(tmp_path, save_steps=100)
    monkeypatch.setattr(mgr, "_build_states", lambda: {})
    monkeypatch.setattr(checkpoint_utils.dcp, "load", lambda *args, **kwargs: None)
    monkeypatch.setattr(checkpoint_utils, "_barrier", lambda: None)

    assert mgr.maybe_resume(resume_from_checkpoint=str(tmp_path / "checkpoint-500")) == 500
    calls = _record_save_calls(mgr)
    mgr.save_final(step=500)

    assert calls == []
