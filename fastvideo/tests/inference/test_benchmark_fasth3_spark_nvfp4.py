# SPDX-License-Identifier: Apache-2.0
"""CPU checks of examples/inference/basic/benchmark_fasth3_spark_nvfp4.py stage accounting."""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_PATH = REPO_ROOT / "examples" / "inference" / "basic" / "benchmark_fasth3_spark_nvfp4.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("benchmark_fasth3_spark_nvfp4", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_decode_seconds_exclude_post_decode_frame_processing() -> None:
    benchmark = _load_benchmark()
    stages = {
        "MiniMaxH3DenoisingStage": 9.0,
        "MiniMaxH3VideoDecodingStage": 2.0,
        "MiniMaxH3AudioDecodingStage": 0.5,
        "PostDecodeFrameProcessStage": 1.0,
    }
    assert benchmark._stage_total(stages, "decod", exclude="postdecode") == 2.5
    assert benchmark._stage_total(stages, "postdecode") == 1.0
    assert benchmark._stage_total(stages, "denois") == 9.0
