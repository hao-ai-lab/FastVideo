# SPDX-License-Identifier: Apache-2.0
"""Generated-output parity for LingBot World 2 causal-fast matrix runs."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path("/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2")
REFERENCE_STATUS = ROOT / "outputs" / "reference" / "status.tsv"
FASTVIDEO_STATUS = ROOT / "outputs" / "fastvideo" / "status.tsv"
COMPARE_SCRIPT = ROOT / "scripts" / "compare_matrix_outputs.py"
COMPARE_JSON = ROOT / "outputs" / "verification" / "matrix_compare.json"


def test_lingbotworld2_generated_matrix_matches_reference_exactly() -> None:
    """Compare generated matrix videos after the source-side launch scripts run."""
    if not REFERENCE_STATUS.exists() or not FASTVIDEO_STATUS.exists():
        pytest.skip("Run lingbot-world-v2/scripts/run_reference_matrix.sh and run_fastvideo_matrix.py first.")

    subprocess.run([sys.executable, str(COMPARE_SCRIPT)], check=True)
    results = json.loads(COMPARE_JSON.read_text(encoding="utf-8"))
    compared = [item for item in results if item.get("comparison") and "exact" in item["comparison"]]
    if not compared:
        pytest.fail("No LingBot World 2 matrix cases had both reference and FastVideo videos to compare.")

    failures = [
        item
        for item in compared
        if not item["comparison"].get("exact")
    ]
    assert failures == []
