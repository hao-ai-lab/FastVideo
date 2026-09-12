# SPDX-License-Identifier: Apache-2.0
"""End-to-end latent parity for Helios-Distilled T2V.

This local test is intentionally heavyweight: it executes the official
Diffusers pipeline and the public FastVideo pipeline with the same CPU RNG,
then compares their final denoised latents. It is not intended for CI without
the pinned local checkpoint.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = REPO_ROOT / "official_weights" / "helios"
RUNNER = REPO_ROOT / "tests" / "local_tests" / "helios" / "run_helios_latent_parity.py"


def _parse_summary(stdout: str) -> dict:
    prefix = "HELIOS_LATENT_PARITY="
    for line in reversed(stdout.splitlines()):
        if line.startswith(prefix):
            return json.loads(line.removeprefix(prefix))
    raise AssertionError(f"Parity runner did not emit {prefix!r}\n{stdout[-4000:]}")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Helios pipeline parity requires CUDA.",
)
def test_helios_distilled_t2v_latent_parity() -> None:
    if not (MODEL_DIR / "model_index.json").is_file():
        pytest.skip(f"Pinned Helios checkpoint not found at {MODEL_DIR}")

    environment = os.environ.copy()
    environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--model-dir",
            str(MODEL_DIR),
            "--gpu",
            "0",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=900,
    )
    summary = _parse_summary(completed.stdout)

    assert summary["shape"] == [1, 16, 9, 16, 24]
    relative_abs_mean_drift = abs(summary["fastvideo_abs_mean"] -
                                  summary["official_abs_mean"]) / summary["official_abs_mean"]

    # Component tests establish exact control flow and close single-forward
    # BF16 parity. This full six-forward DMD loop uses tolerance metrics because
    # fused-QKV and SDPA kernel rounding feeds back through later timesteps.
    assert summary["cosine"] >= 0.95
    assert relative_abs_mean_drift <= 0.05
    assert summary["diff_mean"] <= 0.30
    assert summary["rmse"] <= 0.40
