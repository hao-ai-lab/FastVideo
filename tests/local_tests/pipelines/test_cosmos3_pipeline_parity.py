# SPDX-License-Identifier: Apache-2.0
"""Pinned current-source contracts for the Cosmos3 T2V/I2V pipeline."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from fastvideo.pipelines.stages.cosmos3_stages import (
    cosmos3_arch_invariant_rand,
    cosmos3_format_video_negative_prompt,
    cosmos3_format_video_prompt,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_REVISION = "ed8287fd7477113f8ac4f6b84290514d55cf0cdc"
REFERENCE_REPO = Path(os.getenv("COSMOS3_OFFICIAL_REPO", REPO_ROOT / "cosmos-framework"))

_DURATION_TEMPLATE = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
_RESOLUTION_TEMPLATE = "This video is of {height}x{width} resolution."


def _import_reference_inference():
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

    source_root = REFERENCE_REPO.resolve()
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    module = importlib.import_module("cosmos_framework.inference.inference")
    module_path = Path(module.__file__ or "").resolve()
    assert module_path.is_relative_to(source_root), f"Cosmos3 oracle import escaped pinned repository: {module_path}"
    return module


@pytest.mark.parametrize(
    "prompt",
    [
        "A red panda walks through a bamboo forest.",
        json.dumps({"description": "A red panda walks through a bamboo forest.", "fps": 1}),
    ],
    ids=("plain", "structured"),
)
def test_cosmos3_video_prompt_matches_pinned_official(prompt: str) -> None:
    official = _import_reference_inference()
    kwargs = {
        "fps": 24,
        "num_frames": 189,
        "h": 720,
        "w": 1280,
    }
    prompt_obj = official._parse_json_object_prompt(prompt)
    if prompt_obj is None:
        expected = official._format_prompt_with_template(
            prompt,
            duration_template=_DURATION_TEMPLATE,
            resolution_template=_RESOLUTION_TEMPLATE,
            **kwargs,
        )
    else:
        expected = official._format_json_prompt_with_template(
            prompt_obj,
            aspect_ratio="16,9",
            include_temporal_metadata=True,
            **kwargs,
        )

    actual = cosmos3_format_video_prompt(
        prompt,
        fps=24,
        num_frames=189,
        height=720,
        width=1280,
    )
    assert actual == expected


def test_cosmos3_video_negative_metadata_matches_pinned_official() -> None:
    official = _import_reference_inference()
    prompt = json.dumps({"description": "low quality", "fps": 1})
    expected = official._format_prompt_with_template(
        prompt,
        fps=24,
        num_frames=189,
        duration_template=_DURATION_TEMPLATE,
        resolution_template=_RESOLUTION_TEMPLATE,
        h=720,
        w=1280,
    ).lstrip(".").strip()
    actual = cosmos3_format_video_negative_prompt(
        prompt,
        fps=24,
        num_frames=189,
        height=720,
        width=1280,
    )
    assert actual == expected


@pytest.mark.parametrize("shape", [(16, 2, 4, 6), (48, 48, 45, 80)])
@pytest.mark.parametrize("seed", [0, 1024])
def test_cosmos3_noise_matches_pinned_official(shape: tuple[int, ...], seed: int) -> None:
    _import_reference_inference()
    misc = importlib.import_module("cosmos_framework.utils.misc")
    expected = misc.arch_invariant_rand(shape, torch.float32, "cpu", seed)
    actual = cosmos3_arch_invariant_rand(shape, seed=seed, device=torch.device("cpu"))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cosmos3_registry_exposes_both_video_workloads() -> None:
    from fastvideo.registry import get_registered_models_with_workloads

    for workload in ("t2v", "i2v"):
        registered = get_registered_models_with_workloads(workload)
        assert any(model["id"] == "nvidia/Cosmos3-Nano" for model in registered)
