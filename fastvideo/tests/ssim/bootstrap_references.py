# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import pytest

import fastvideo.envs as envs
from fastvideo.tests.ssim.reference_utils import get_output_quality_tier
from fastvideo.tests.ssim.reference_videos_cli import (
    upload_draft_reference_artifact, )

TRUE_VALUES = {"1", "true", "yes", "on"}


def bootstrap_mode_enabled() -> bool:
    return os.environ.get("FASTVIDEO_SSIM_BOOTSTRAP_MODE", "").strip().lower() in TRUE_VALUES


def xfail_missing_reference_in_bootstrap_mode(
    *,
    generated_artifact_path: str,
    reference_folder: str,
    artifact_kind: str,
) -> None:
    if not bootstrap_mode_enabled():
        return

    generated_path = Path(generated_artifact_path)
    if not generated_path.exists():
        raise FileNotFoundError(
            f"SSIM bootstrap mode is enabled, but generated {artifact_kind} artifact is missing: {generated_path}")

    repo_id = envs.FASTVIDEO_TEST_SSIM_REFERENCE_HF_REPO.get()
    repo_type = envs.FASTVIDEO_TEST_SSIM_REFERENCE_HF_REPO_TYPE.get()
    draft_path = upload_draft_reference_artifact(
        repo_id=repo_id,
        repo_type=repo_type,
        generated_artifact_path=generated_path,
        reference_folder=Path(reference_folder),
    )
    pytest.xfail("SSIM bootstrap mode generated a draft "
                 f"{artifact_kind} reference at {repo_id}/{draft_path}. "
                 "Review it, then promote with "
                 "`python fastvideo/tests/ssim/reference_videos_cli.py promote-draft "
                 f"--quality-tier {get_output_quality_tier()} --model-id <model_id>`.")
