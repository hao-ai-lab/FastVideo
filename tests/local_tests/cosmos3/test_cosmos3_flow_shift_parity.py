# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 per-modality UniPC shift parity with the pinned framework."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from fastvideo.pipelines.stages.cosmos3_stages import Cosmos3DenoisingStage

pytestmark = [pytest.mark.local]

REPO_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_REVISION = "ed8287fd7477113f8ac4f6b84290514d55cf0cdc"
REFERENCE_REPO = Path(os.getenv("COSMOS3_OFFICIAL_REPO", REPO_ROOT / "cosmos-framework"))


def _official_shift(mode: str) -> float:
    if not REFERENCE_REPO.is_dir():
        pytest.skip(f"Pinned NVIDIA cosmos-framework clone not found: {REFERENCE_REPO}")
    result = subprocess.run(
        ["git", "-C", str(REFERENCE_REPO), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = result.stdout.strip()
    assert actual == REFERENCE_REVISION, f"Cosmos3 oracle must be pinned at {REFERENCE_REVISION}, got {actual}"

    defaults_path = REFERENCE_REPO / "cosmos_framework" / "inference" / "defaults" / mode / "sample_args.json"
    defaults = json.loads(defaults_path.read_text(encoding="utf-8"))
    return float(defaults["shift"])


@pytest.mark.parametrize(
    ("mode", "is_video"),
    [
        pytest.param("text2video", True, id="t2v"),
        pytest.param("image2video", True, id="i2v"),
        pytest.param("text2image", False, id="t2i"),
    ],
)
def test_flow_shift_matches_official_modality_default(mode: str, is_video: bool) -> None:
    expected = _official_shift(mode)
    actual = Cosmos3DenoisingStage._flow_shift_for_mode(is_video=is_video)
    assert actual == expected
