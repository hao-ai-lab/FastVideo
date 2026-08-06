# SPDX-License-Identifier: Apache-2.0
"""Strict stage and end-to-end parity for the MAGI-2 release pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_ROOT = Path(
    os.environ.get("MAGI2_OFFICIAL_REF_DIR", REPO_ROOT.parent / "MAGI-2-preview")
)
WEIGHTS_ROOT = Path(
    os.environ.get("MAGI2_LOCAL_WEIGHTS_DIR", REPO_ROOT / "official_weights" / "magi2")
)
CONVERTED_ROOT = Path(
    os.environ.get("MAGI2_CONVERTED_WEIGHTS_DIR", REPO_ROOT / "converted_weights" / "magi2")
)
WORKER_PATH = Path(__file__).with_name("_pipeline_parity_worker.py")
CAPTURE_ROOT = REPO_ROOT / "archived" / "magi2_parity" / "validation" / "pipeline"
OFFICIAL_REVISION = "073c84f2102ec3c9287623113a103c14402770ad"
WORLD_SIZE = 8


pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < WORLD_SIZE,
    reason=f"MAGI-2 pipeline numerical parity requires {WORLD_SIZE} CUDA devices",
)


def _require_parity_sources() -> None:
    """Require clean pinned source trees and complete converted components."""
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=OFFICIAL_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert revision == OFFICIAL_REVISION
    official_changes = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=OFFICIAL_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert not official_changes, (
        f"Official MAGI-2 checkout must be clean for parity:\n{official_changes}"
    )
    required_paths = (
        WEIGHTS_ROOT / "preview" / "model.safetensors.index.json",
        WEIGHTS_ROOT / "refiner" / "model.safetensors.index.json",
        WEIGHTS_ROOT / "turbo_vae" / "checkpoint.ckpt",
        CONVERTED_ROOT / "model_index.json",
        CONVERTED_ROOT / "transformer" / "model.safetensors.index.json",
        CONVERTED_ROOT / "transformer_2" / "model.safetensors.index.json",
    )
    for required_path in required_paths:
        assert required_path.is_file(), f"MAGI-2 parity input is missing: {required_path}"


def _run_implementation(implementation: str) -> Path:
    """Launch one full release-profile implementation in an isolated torchrun job."""
    output_dir = CAPTURE_ROOT / implementation
    environment = os.environ.copy()
    environment.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "MAGI2_CKPT_ROOT": str(WEIGHTS_ROOT),
            "MAGI2_CONVERTED_WEIGHTS_DIR": str(CONVERTED_ROOT),
            "MAGI2_DETERMINISTIC": "1",
            "MAGI2_LOCAL_WEIGHTS_DIR": str(WEIGHTS_ROOT),
            "MAGI2_OFFICIAL_REF_DIR": str(OFFICIAL_ROOT),
            "MAGI2_TEXT_ENC_OFFLOAD_MODE": "roundtrip",
            "MAGI2_VAE_OFFLOAD_MODE": "roundtrip",
            "MAGI_ATTENTION_DETERMINISTIC_MODE": "1",
            "MAGI_COMPILE_COMPILE_MODE": "NONE",
            "OMP_NUM_THREADS": "1",
            "PYTHONHASHSEED": "42",
        }
    )
    environment.pop("MAGI2_SAVE_LATENT_PATH", None)
    environment.pop("NEGATIVE_PROMPT", None)
    environment.pop("SKIP_LOAD_MODEL", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={WORLD_SIZE}",
            str(WORKER_PATH),
            "--implementation",
            implementation,
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=21600,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"{implementation} MAGI-2 pipeline torchrun failed.\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return output_dir


def _load_capture(output_dir: Path) -> dict[str, Any]:
    """Load and validate one stage-digest manifest."""
    artifact_path = output_dir / "capture.json"
    assert artifact_path.is_file(), f"MAGI-2 capture is missing: {artifact_path}"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["schema_version"] == 1
    assert artifact["world_size"] == WORLD_SIZE
    assert artifact["preview_steps"] == 100
    assert artifact["refiner_steps"] == 5
    assert artifact["cases"].keys() == {"t2v", "i2v"}
    return artifact


def _first_difference(
    fastvideo_value: Any,
    official_value: Any,
    path: str,
) -> str | None:
    """Return the first structural or numerical difference in two manifests."""
    if type(fastvideo_value) is not type(official_value):
        return (
            f"{path}: type differs: {type(fastvideo_value).__name__} versus "
            f"{type(official_value).__name__}"
        )
    if isinstance(official_value, dict):
        if fastvideo_value.keys() != official_value.keys():
            return (
                f"{path}: keys differ: {sorted(fastvideo_value)} versus "
                f"{sorted(official_value)}"
            )
        for key in official_value:
            difference = _first_difference(
                fastvideo_value[key],
                official_value[key],
                f"{path}.{key}",
            )
            if difference is not None:
                return difference
        return None
    if isinstance(official_value, list):
        if len(fastvideo_value) != len(official_value):
            return (
                f"{path}: list length differs: {len(fastvideo_value)} versus "
                f"{len(official_value)}"
            )
        for index, (fastvideo_item, official_item) in enumerate(
            zip(fastvideo_value, official_value, strict=True)
        ):
            difference = _first_difference(
                fastvideo_item,
                official_item,
                f"{path}[{index}]",
            )
            if difference is not None:
                return difference
        return None
    if fastvideo_value != official_value:
        return f"{path}: {fastvideo_value!r} differs from {official_value!r}"
    return None


def test_magi2_pipeline_matches_official_release_exactly() -> None:
    """Match T2V and I2V stage tensors, decoded video, and decoded audio exactly."""
    _require_parity_sources()
    official_capture = _load_capture(_run_implementation("official"))
    fastvideo_capture = _load_capture(_run_implementation("fastvideo"))
    assert official_capture["implementation"] == "official"
    assert fastvideo_capture["implementation"] == "fastvideo"
    difference = _first_difference(
        fastvideo_capture["cases"],
        official_capture["cases"],
        "cases",
    )
    assert difference is None, difference
