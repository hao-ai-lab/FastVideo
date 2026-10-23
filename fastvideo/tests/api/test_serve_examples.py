# SPDX-License-Identifier: Apache-2.0
"""Shipped serve examples must parse through the production CLI path."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from fastvideo.entrypoints.cli.inference_config import build_serve_config

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SERVE_CONFIGS = sorted((_REPO_ROOT / "examples" / "serving").glob("openai_*.yaml"))


@pytest.mark.parametrize("config_path", _SERVE_CONFIGS, ids=lambda path: path.name)
def test_serve_example_parses(config_path: Path) -> None:
    config = build_serve_config(argparse.Namespace(config=str(config_path)))
    assert config.generator.model_path
    assert config.generator.engine.num_gpus > 0
