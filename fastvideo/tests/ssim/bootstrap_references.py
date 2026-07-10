# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import pytest

from fastvideo.tests.ssim.reference_utils import get_output_quality_tier
from fastvideo.tests.ssim.reference_videos_cli import (
    BOOTSTRAP_ENV_KEY,
    DEFAULT_REPO_ID,
    DEFAULT_REPO_TYPE,
    HF_REPO_ENV_KEY,
    HF_REPO_TYPE_ENV_KEY,
    upload_draft_reference_artifact,
)

TRUE_VALUES = {"1", "true", "yes", "on"}


def bootstrap_mode_enabled() -> bool:
    return os.environ.get(BOOTSTRAP_ENV_KEY, "").strip().lower() in TRUE_VALUES


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
            f"SSIM bootstrap mode is enabled, but generated {artifact_kind} artifact is missing: {generated_path}"
        )

    repo_id = os.environ.get(HF_REPO_ENV_KEY, DEFAULT_REPO_ID)
    repo_type = os.environ.get(HF_REPO_TYPE_ENV_KEY, DEFAULT_REPO_TYPE)
    draft_path = upload_draft_reference_artifact(
        repo_id=repo_id,
        repo_type=repo_type,
        generated_artifact_path=generated_path,
        reference_folder=Path(reference_folder),
    )
    pytest.xfail(
        "SSIM bootstrap mode generated a draft "
        f"{artifact_kind} reference at {repo_id}/{draft_path}. "
        "Review it, then promote with "
        "`python fastvideo/tests/ssim/reference_videos_cli.py promote-draft "
        f"--quality-tier {get_output_quality_tier()} --model-id <model_id>`."
    )
