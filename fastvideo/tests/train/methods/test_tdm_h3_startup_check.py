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
from fastvideo.train.models.base import ModelBase
from fastvideo.train.models.minimax_h3 import MiniMaxH3Model
from fastvideo.train.models.minimax_h3.minimax_h3 import (
    _AUDIO_SCHEDULER_SHIFT,
    _VIDEO_SCHEDULER_SHIFT,
)
from fastvideo.train.utils.config import load_run_config
from fastvideo.train.utils.training_config import (
    DataConfig,
    OptimizerConfig,
    TrainingConfig,
    TrainingLoopConfig,
)

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
    ModelBase.__init__(student)
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


class _FakeH3Role(MiniMaxH3Model):
    """H3 role with real scheduler shifts but no transformer weights."""

    def __init__(self, *, video: float = 12.0, audio: float = 3.0, trainable: bool = True) -> None:
        ModelBase.__init__(self, trainable=trainable)
        self.transformer = torch.nn.Linear(1, 1, bias=False)
        self.transformer.requires_grad_(trainable)
        self.noise_scheduler = MiniMaxH3Scheduler(shift=video)
        self.audio_noise_scheduler = MiniMaxH3Scheduler(shift=audio)

    def init_preprocessors(self, training_config: TrainingConfig) -> None:
        pass


def _build_h3_method(model_path: str, *, video: float = 12.0, audio: float = 3.0) -> TDMMethod:
    training = TrainingConfig(
        data=DataConfig(preprocessed_data_type="text_only", seed=0),
        optimizer=OptimizerConfig(
            learning_rate=1.0e-3,
            betas=(0.0, 0.999),
            weight_decay=0.0,
            lr_scheduler="constant",
        ),
        loop=TrainingLoopConfig(max_train_steps=4),
        model_path=model_path,
    )
    cfg = SimpleNamespace(
        training=training,
        method={
            "rollout_mode": "simulate",
            "generator_update_interval": 1,
            "real_score_guidance_scale": 1.0,
            "tdm_denoising_steps": [1000, 750, 500, 250],
            "student_sample_type": "ode",
            "noise_interval_mode": "separate",
            "use_randmid": False,
            "max_grad_norm": 1.0,
            "fake_score_learning_rate": 1.0e-3,
            "fake_score_betas": [0.0, 0.999],
            "fake_score_lr_scheduler": "constant",
        },
        validation={},
    )
    return TDMMethod(
        cfg=cfg,
        role_models={
            "student": _FakeH3Role(video=video, audio=audio, trainable=True),
            "teacher": _FakeH3Role(video=video, audio=audio, trainable=False),
            "critic": _FakeH3Role(video=video, audio=audio, trainable=True),
        },
    )


def test_real_h3_student_queries_both_scheduler_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_scheduler_configs(tmp_path)
    queries: list[tuple[str, bool]] = []
    read_config = tdm_module.read_optional_model_json

    def spy(model_path: str, owner: str, revision: str | None = None, *, strict: bool = False) -> dict | None:
        queries.append((owner, strict))
        return read_config(model_path, owner, revision, strict=strict)

    monkeypatch.setattr(tdm_module, "read_optional_model_json", spy)

    _check(str(tmp_path))

    assert queries == [
        ("scheduler/scheduler_config.json", True),
        ("audio_scheduler/scheduler_config.json", True),
    ]


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

    assert any("has no audio_scheduler/scheduler_config.json" in warning for warning in warnings)


def test_scheduler_config_without_shift_warns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(tdm_module.logger, "warning", lambda message, *args: warnings.append(message % args))
    _write_scheduler_configs(tmp_path)
    (tmp_path / "audio_scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")

    _check(str(tmp_path), audio=4.0)

    assert any("has no shift" in warning and "audio_scheduler/scheduler_config.json" in warning
               for warning in warnings)


def test_student_without_h3_shift_attributes_fails_with_a_named_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NoShiftStudent(torch.nn.Module):
        pass

    method = TDMMethod.__new__(TDMMethod)
    torch.nn.Module.__init__(method)
    method.training_config = SimpleNamespace(model_path="org/repo")
    method.student = _NoShiftStudent()
    monkeypatch.setattr(TDMMethod, "_model_family", lambda self: "minimax_h3")
    monkeypatch.setattr(tdm_module, "read_optional_model_json", lambda *args, **kwargs: {"shift": 12.0})

    with pytest.raises(ValueError, match="exposes no video_scheduler_shift"):
        method._validate_h3_checkpoint_shifts()


def test_nonexistent_local_model_path_fails_before_hub_lookup(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="training.model_path"):
        _check(str(tmp_path / "missing"))


def test_relative_local_model_path_reports_hub_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from huggingface_hub.utils import HFValidationError

    def fail(model_path: str, owner: str, revision: str | None = None, *, strict: bool = False) -> dict | None:
        del owner, revision, strict
        raise HFValidationError(f"Repo id must be in the form 'repo_name': {model_path!r}")

    monkeypatch.setattr(tdm_module, "read_optional_model_json", fail)

    with pytest.raises(ValueError, match="training.model_path.*local H3 checkpoint"):
        _check("checkpoints/does-not-exist")


def test_offline_hub_failure_is_not_reported_as_a_missing_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from huggingface_hub.utils import LocalEntryNotFoundError

    def fail(model_path: str, owner: str, revision: str | None = None, *, strict: bool = False) -> dict | None:
        del model_path, owner, revision, strict
        raise LocalEntryNotFoundError("Cannot find the requested files in the disk cache")

    monkeypatch.setattr(tdm_module, "read_optional_model_json", fail)

    with pytest.raises(ValueError, match="could not fetch"):
        _check("org/repo")


def test_constructor_strict_helper_reports_an_unreachable_hub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from huggingface_hub.utils import LocalEntryNotFoundError

    monkeypatch.setattr(TDMMethod, "_model_family", lambda self: "minimax_h3")

    def fail(**kwargs) -> str:
        raise LocalEntryNotFoundError(f"{kwargs['filename']} is not in the disk cache")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fail)

    with pytest.raises(ValueError, match="could not fetch"):
        _build_h3_method("org/repo")


def test_gated_hub_failure_asks_for_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from huggingface_hub.errors import GatedRepoError

    class _GatedRepoError(GatedRepoError):

        def __init__(self, message: str) -> None:
            Exception.__init__(self, message)

    def fail(**kwargs) -> str:
        raise _GatedRepoError(f"{kwargs['filename']} is in a gated repo")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fail)

    with pytest.raises(ValueError, match="HF_TOKEN"):
        _check("MiniMaxAI/MiniMax-H3")


def test_missing_hub_repo_suggests_both_local_and_remote_fixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from huggingface_hub.utils import RepositoryNotFoundError

    class _MissingRepoError(RepositoryNotFoundError):

        def __init__(self, message: str) -> None:
            Exception.__init__(self, message)

    def fail(model_path: str, owner: str, revision: str | None = None, *, strict: bool = False) -> dict | None:
        del model_path, owner, revision, strict
        raise _MissingRepoError("404 Client Error: repository not found")

    monkeypatch.setattr(tdm_module, "read_optional_model_json", fail)

    with pytest.raises(ValueError, match=r"existing local H3 checkpoint.*correct Hub repo id"):
        _check("checkpoints/does-not-exist")


@pytest.mark.parametrize("status_code", [401, 403])
def test_file_download_rejected_credentials_ask_for_a_token(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    # A 401/403 HfHubHTTPError raised directly by hf_hub_download. huggingface_hub
    # re-raises a HEAD 401 unchanged, while a HEAD 403 is normally wrapped into
    # LocalEntryNotFoundError (see test_head_403_is_reported_as_a_token_problem).
    from huggingface_hub.utils import HfHubHTTPError

    class _UnauthorizedError(HfHubHTTPError):

        def __init__(self, message: str) -> None:
            Exception.__init__(self, message)
            self.response = SimpleNamespace(status_code=status_code)

    def fail(**kwargs) -> str:
        raise _UnauthorizedError(f"{kwargs['filename']}: {status_code} Client Error")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fail)

    with pytest.raises(ValueError, match="HF_TOKEN"):
        _check("org/repo")


def test_head_403_is_reported_as_a_token_problem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # huggingface_hub converts a non-401 HEAD HfHubHTTPError (including 403)
    # into LocalEntryNotFoundError chained from the original error, so the
    # connectivity branch must inspect the cause to keep the token advice.
    from huggingface_hub.utils import HfHubHTTPError, LocalEntryNotFoundError

    class _ForbiddenError(HfHubHTTPError):

        def __init__(self, message: str) -> None:
            Exception.__init__(self, message)
            self.response = SimpleNamespace(status_code=403)

    def fail(**kwargs) -> str:
        try:
            raise _ForbiddenError("403 Client Error: Forbidden")
        except _ForbiddenError as cause:
            raise LocalEntryNotFoundError(f"{kwargs['filename']} is not in the disk cache") from cause

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fail)

    with pytest.raises(ValueError, match="HF_TOKEN"):
        _check("MiniMaxAI/MiniMax-H3")


def test_head_403_for_a_disabled_repo_does_not_ask_for_a_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A disabled repo is a 403 HfHubHTTPError that huggingface_hub wraps in
    # LocalEntryNotFoundError, but no token can open it.
    from huggingface_hub.utils import DisabledRepoError, LocalEntryNotFoundError

    class _DisabledRepoError(DisabledRepoError):

        def __init__(self, message: str) -> None:
            Exception.__init__(self, message)
            self.response = SimpleNamespace(status_code=403)

    def fail(**kwargs) -> str:
        try:
            raise _DisabledRepoError("403 Client Error: Access to this resource is disabled.")
        except _DisabledRepoError as cause:
            raise LocalEntryNotFoundError(f"{kwargs['filename']} is not in the disk cache") from cause

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fail)

    with pytest.raises(ValueError, match="disabled") as excinfo:
        _check("MiniMaxAI/MiniMax-H3")

    assert "HF_TOKEN" not in str(excinfo.value)


def test_direct_403_for_a_disabled_repo_does_not_ask_for_a_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Direct (GET-stage) DisabledRepoError; the HEAD path wraps it in
    # LocalEntryNotFoundError, covered by the previous test.
    from huggingface_hub.utils import DisabledRepoError

    class _DisabledRepoError(DisabledRepoError):

        def __init__(self, message: str) -> None:
            Exception.__init__(self, message)
            self.response = SimpleNamespace(status_code=403)

    def fail(**kwargs) -> str:
        raise _DisabledRepoError(f"{kwargs['filename']}: Access to this resource is disabled.")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fail)

    with pytest.raises(ValueError, match="disabled") as excinfo:
        _check("MiniMaxAI/MiniMax-H3")

    assert "HF_TOKEN" not in str(excinfo.value)


def test_constructor_rejects_a_mismatched_h3_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TDMMethod, "_model_family", lambda self: "minimax_h3")
    _write_scheduler_configs(tmp_path)

    with pytest.raises(ValueError, match="video_scheduler_shift="):
        _build_h3_method(str(tmp_path), video=10.0)


def test_constructor_accepts_a_matching_h3_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TDMMethod, "_model_family", lambda self: "minimax_h3")
    _write_scheduler_configs(tmp_path)

    method = _build_h3_method(str(tmp_path))

    assert method.student.video_scheduler_shift == 12.0
    assert method.student.audio_scheduler_shift == 3.0


def test_shipped_h3_overfit_config_uses_the_checkpoint_defaults(tmp_path: Path) -> None:
    cfg = load_run_config(_H3_TDM_CONFIG, ["--training.model_path", str(tmp_path)])
    pipeline_config = cfg.training.pipeline_config
    assert pipeline_config.video_scheduler_shift in (None, _VIDEO_SCHEDULER_SHIFT)
    assert pipeline_config.audio_scheduler_shift in (None, _AUDIO_SCHEDULER_SHIFT)
    _write_scheduler_configs(tmp_path, video=_VIDEO_SCHEDULER_SHIFT, audio=_AUDIO_SCHEDULER_SHIFT)

    _check(cfg.training.model_path, video=_VIDEO_SCHEDULER_SHIFT, audio=_AUDIO_SCHEDULER_SHIFT)

    assert cfg.training.model_path == str(tmp_path)
