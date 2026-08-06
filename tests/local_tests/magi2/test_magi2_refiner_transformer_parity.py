# SPDX-License-Identifier: Apache-2.0
"""Strict eight-GPU numerical parity for the MAGI-2 refiner transformer.

Coverage scope: both. The FastVideo side uses the production checkpoint loader,
and the captures compare every implementation boundary that owns numerical
behavior from the packed proxy input through the depacked video and audio.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_ROOT = Path(
    os.environ.get("MAGI2_OFFICIAL_REF_DIR", REPO_ROOT.parent / "MAGI-2-preview")
)
WEIGHTS_ROOT = Path(
    os.environ.get("MAGI2_LOCAL_WEIGHTS_DIR", REPO_ROOT / "official_weights" / "magi2")
)
WORKER_PATH = Path(__file__).with_name("_refiner_transformer_parity_worker.py")
CAPTURE_ROOT = REPO_ROOT / "archived" / "magi2_parity" / "validation" / "refiner_transformer"
OFFICIAL_REVISION = "073c84f2102ec3c9287623113a103c14402770ad"
WORLD_SIZE = 8


def _require_parity_sources() -> None:
    """Require eight GPUs, the pinned source, and every refiner weight shard."""
    if torch.cuda.device_count() < WORLD_SIZE:
        raise AssertionError(
            f"MAGI-2 refiner numerical parity requires {WORLD_SIZE} CUDA devices; "
            f"found {torch.cuda.device_count()}"
        )
    model_path = OFFICIAL_ROOT / "inference" / "model" / "magi2_refiner.py"
    if not model_path.is_file():
        raise AssertionError(f"Official MAGI-2 refiner source is missing: {model_path}")
    index_path = WEIGHTS_ROOT / "refiner" / "model.safetensors.index.json"
    if not index_path.is_file():
        raise AssertionError(f"MAGI-2 refiner checkpoint index is missing: {index_path}")
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    missing_shards = sorted(
        shard_name
        for shard_name in set(weight_map.values())
        if not (index_path.parent / shard_name).is_file()
    )
    if missing_shards:
        raise AssertionError(f"MAGI-2 refiner checkpoint shards are missing: {missing_shards}")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=OFFICIAL_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert revision == OFFICIAL_REVISION
    checkout_changes = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=OFFICIAL_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert not checkout_changes, (
        "Official MAGI-2 checkout contains local changes:\n" + checkout_changes
    )


def _run_implementation(implementation: str) -> Path:
    """Launch one implementation in an isolated eight-rank torchrun job."""
    output_dir = CAPTURE_ROOT / implementation
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "MAGI2_CKPT_ROOT": str(WEIGHTS_ROOT),
            "MAGI2_DETERMINISTIC": "1",
            "MAGI2_LOCAL_WEIGHTS_DIR": str(WEIGHTS_ROOT),
            "MAGI2_OFFICIAL_REF_DIR": str(OFFICIAL_ROOT),
            "MAGI_ATTENTION_DETERMINISTIC_MODE": "1",
            "MAGI_COMPILE_COMPILE_MODE": "NONE",
            "OMP_NUM_THREADS": "1",
            "PYTHONHASHSEED": "42",
        }
    )
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
        timeout=7200,
    )
    log_path = CAPTURE_ROOT / f"{implementation}.log"
    log_path.write_text(
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"{implementation} refiner transformer torchrun failed; see {log_path}.\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return output_dir


def _assert_exact(actual: Any, expected: Any, path: str) -> None:
    """Recursively require identical capture structure, metadata, and values."""
    assert type(actual) is type(expected), (
        f"{path}: type differs: {type(actual).__name__} versus "
        f"{type(expected).__name__}"
    )
    if isinstance(expected, torch.Tensor):
        assert actual.shape == expected.shape, path
        assert actual.dtype == expected.dtype, path
        assert actual.stride() == expected.stride(), path
        if not torch.equal(actual, expected):
            difference = (actual.float() - expected.float()).abs()
            raise AssertionError(
                f"{path}: tensor values differ; mismatched="
                f"{torch.count_nonzero(actual != expected).item()}, "
                f"max_abs={difference.max().item()}"
            )
        return
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            _assert_exact(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for index, (actual_value, expected_value) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _assert_exact(actual_value, expected_value, f"{path}[{index}]")
        return
    assert actual == expected, path


def _load_rank_capture(output_dir: Path, rank: int) -> dict[str, Any]:
    """Load one rank artifact and validate its fixed capture envelope."""
    artifact_path = output_dir / f"rank_{rank}.pt"
    if not artifact_path.is_file():
        raise AssertionError(f"Refiner parity capture is missing: {artifact_path}")
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    assert artifact["schema_version"] == 1
    assert artifact["rank"] == rank
    assert artifact["world_size"] == WORLD_SIZE
    assert set(artifact) == {
        "schema_version",
        "implementation",
        "rank",
        "world_size",
        "case",
    }
    return artifact


def test_magi2_refiner_transformer_matches_official_exactly() -> None:
    """Match every refiner tensor boundary through video and audio depacking."""
    _require_parity_sources()
    official_output_dir = _run_implementation("official")
    fastvideo_output_dir = _run_implementation("fastvideo")
    for rank in range(WORLD_SIZE):
        official_capture = _load_rank_capture(official_output_dir, rank)
        fastvideo_capture = _load_rank_capture(fastvideo_output_dir, rank)
        assert official_capture["implementation"] == "official"
        assert fastvideo_capture["implementation"] == "fastvideo"
        _assert_exact(
            fastvideo_capture["case"],
            official_capture["case"],
            f"rank_{rank}.case",
        )
