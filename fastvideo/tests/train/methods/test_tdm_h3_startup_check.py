# SPDX-License-Identifier: Apache-2.0
"""H3 TDM fails before training when its shifts disagree with the checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from fastvideo.train.methods.distribution_matching import tdm as tdm_module
from fastvideo.train.methods.distribution_matching.tdm import TDMMethod
from fastvideo.train.models.minimax_h3 import MiniMaxH3Model
from fastvideo.train.utils.config import load_run_config

_H3_TDM_CONFIG = "examples/train/configs/distribution_matching/overfit_minimax_h3_t2va_tdm.yaml"


def _write_scheduler_configs(
    root: Path,
    *,
    video: float | None = 12.0,
    audio: float | None = 3.0,
) -> None:
    for owner, shift in (("scheduler", video), ("audio_scheduler", audio)):
        if shift is None:
            continue
        directory = root / owner
        directory.mkdir()
        (directory / "scheduler_config.json").write_text(json.dumps({"shift": shift}), encoding="utf-8")


def _h3_student(*, video: float = 12.0, audio: float = 3.0) -> MiniMaxH3Model:
    student = MiniMaxH3Model.__new__(MiniMaxH3Model)
    torch.nn.Module.__init__(student)
    student.noise_scheduler = MiniMaxH3Scheduler(shift=video)
    student.audio_noise_scheduler = MiniMaxH3Scheduler(shift=audio)
    return student


def _check(model_path: str, *, video: float = 12.0, audio: float = 3.0) -> None:
    method = TDMMethod.__new__(TDMMethod)
    torch.nn.Module.__init__(method)
    method.training_config = SimpleNamespace(model_path=model_path)
    method.student = _h3_student(video=video, audio=audio)
    assert method._model_family() == "minimax_h3"
    method._validate_h3_checkpoint_shifts()


def test_real_h3_student_queries_both_scheduler_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_scheduler_configs(tmp_path)
    queries: list[str] = []
    read_config = tdm_module.read_optional_model_json

    def spy(model_path: str, owner: str, revision: str | None = None) -> dict | None:
        queries.append(owner)
        return read_config(model_path, owner, revision)

    monkeypatch.setattr(tdm_module, "read_optional_model_json", spy)

    _check(str(tmp_path))

    assert queries == ["scheduler/scheduler_config.json", "audio_scheduler/scheduler_config.json"]


@pytest.mark.parametrize(("video", "audio", "attribute"), [
    (10.0, 3.0, "video_scheduler_shift"),
    (12.0, 4.0, "audio_scheduler_shift"),
])
def test_real_h3_student_rejects_shift_mismatch(
    tmp_path: Path,
    video: float,
    audio: float,
    attribute: str,
) -> None:
    _write_scheduler_configs(tmp_path)

    with pytest.raises(ValueError, match=f"{attribute}="):
        _check(str(tmp_path), video=video, audio=audio)


def test_missing_scheduler_config_warns_instead_of_skipping_silently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(tdm_module.logger, "warning", lambda message, *args: warnings.append(message % args))
    _write_scheduler_configs(tmp_path)
    (tmp_path / "audio_scheduler" / "scheduler_config.json").unlink()

    _check(str(tmp_path), audio=4.0)

    assert any("audio_scheduler/scheduler_config.json" in warning for warning in warnings)


def test_scheduler_config_without_shift_warns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(tdm_module.logger, "warning", lambda message, *args: warnings.append(message % args))
    _write_scheduler_configs(tmp_path)
    (tmp_path / "audio_scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")

    _check(str(tmp_path), audio=4.0)

    assert any("audio_scheduler/scheduler_config.json" in warning for warning in warnings)


def test_nonexistent_local_model_path_fails_before_hub_lookup(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="training.model_path"):
        _check(str(tmp_path / "missing"))


def test_shipped_h3_overfit_config_accepts_a_matching_local_checkpoint(tmp_path: Path) -> None:
    cfg = load_run_config(_H3_TDM_CONFIG, ["--training.model_path", str(tmp_path)])
    _write_scheduler_configs(tmp_path)

    _check(cfg.training.model_path)

    assert cfg.training.model_path == str(tmp_path)
