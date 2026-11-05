# SPDX-License-Identifier: Apache-2.0
"""A Ref2VA PDD student's export carries the fastvideo_inference.json that FastH3 inference reads back."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fastvideo import envs
from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.tests.stages.test_minimax_h3_pdd_contract import GRID32_BLOCKS8, PDD_FILE
from fastvideo.train.models.minimax_h3.minimax_h3_ref2v_pdd import MiniMaxH3Ref2VPDDModel

BASE_PIN = "9bfb6693f2cf6de171db46d1aa586f67d773a1da"


@pytest.fixture(autouse=True)
def _unset_backend_env_var(env_overrides):
    env_overrides.enter_context(envs.FASTVIDEO_ATTENTION_BACKEND.override(None))


def _write_json(root: Path, files: dict[str, dict]) -> Path:
    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content), encoding="utf-8")
    return root


def _base(root: Path, *, video_shift: float = 12.0, contract: dict | None = None) -> Path:
    files = {
        "scheduler/scheduler_config.json": {"shift": video_shift},
        "audio_scheduler/scheduler_config.json": {"shift": 3.0},
    }
    if contract is not None:
        files["fastvideo_inference.json"] = contract
    return _write_json(root, files)


def _student(init_from: Path, *, backend=AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3, revision: str | None = None,
             video_shift: float = 12.0) -> MiniMaxH3Ref2VPDDModel:
    """The B300 OmniRef recipe's student: Grid32, VSA90 at tile 128, reference keep rate 0.1, clocks 12/3."""
    student = MiniMaxH3Ref2VPDDModel.__new__(MiniMaxH3Ref2VPDDModel)
    student._init_from = str(init_from)
    student.pdd_steps = 32
    student.attention_backend = backend
    student.noise_scheduler = MiniMaxH3Scheduler(shift=video_shift)
    student.audio_noise_scheduler = MiniMaxH3Scheduler(shift=3.0)
    student.training_config = SimpleNamespace(
        vsa_sparsity=0.9,
        vsa_ref_keep_rate=0.1,
        vsa_tile_size=128,
        checkpoint=SimpleNamespace(inference_base_model_revision=revision),
    )
    return student


def test_hf_snapshot_student_exports_the_contract_inference_reads_back(tmp_path: Path) -> None:
    base = _base(tmp_path / "hub" / "models--MiniMaxAI--MiniMax-H3" / "snapshots" / BASE_PIN)

    contract = _student(base).inference_contract(pdd_step_indices=GRID32_BLOCKS8)

    # The generic writer reproduces the OmniRef PDD-8 export's file for its recipe.
    assert contract == PDD_FILE
    export = _write_json(tmp_path / "export", {
        "transformer_ref/config.json": {"pdd_steps": 32},
        "scheduler/scheduler_config.json": {"shift": 12.0},
        "audio_scheduler/scheduler_config.json": {"shift": 3.0},
        "fastvideo_inference.json": contract,
    })
    args = FastVideoArgs(model_path=str(export), pipeline_config=MiniMaxH3PipelineConfig())
    assert args.pipeline_config.pdd_step_indices == tuple(GRID32_BLOCKS8)
    assert (args.attention_backend, args.VSA_tile_size, args.VSA_sparsity) == ("VIDEO_SPARSE_ATTN_H3", 128, 0.9)
    assert args.pipeline_config.vsa_ref_keep_rate == 0.1


def test_pdd_warm_start_keeps_the_base_it_was_distilled_against(tmp_path: Path) -> None:
    warm_start = _base(tmp_path / "pdd-1250", contract=PDD_FILE)
    contract = _student(warm_start).inference_contract(pdd_step_indices=GRID32_BLOCKS8)
    assert contract is not None and contract["base_model_revision"] == PDD_FILE["base_model_revision"]


def test_configured_base_model_revision_takes_precedence(tmp_path: Path) -> None:
    student = _student(_base(tmp_path / "local"), revision="hf://example/MiniMax-H3@0123")
    contract = student.inference_contract(pdd_step_indices=GRID32_BLOCKS8)
    assert contract is not None and contract["base_model_revision"] == "hf://example/MiniMax-H3@0123"


def test_plain_local_base_needs_an_explicit_revision(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="training.checkpoint.inference_base_model_revision"):
        _student(_base(tmp_path / "local")).inference_contract(pdd_step_indices=GRID32_BLOCKS8)


def test_training_clock_must_match_the_exported_scheduler(tmp_path: Path) -> None:
    student = _student(_base(tmp_path / "local"), revision="hf://example/MiniMax-H3@0123", video_shift=10.0)
    with pytest.raises(ValueError, match="video_scheduler_shift=10.0"):
        student.inference_contract(pdd_step_indices=GRID32_BLOCKS8)


def test_dense_student_exports_without_a_contract(tmp_path: Path) -> None:
    student = _student(_base(tmp_path / "local"), backend=AttentionBackendEnum.FLASH_ATTN)
    assert student.inference_contract(pdd_step_indices=GRID32_BLOCKS8) is None
