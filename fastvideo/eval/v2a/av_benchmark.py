# SPDX-License-Identifier: Apache-2.0
"""Isolated launcher for the official ``hkchengrex/av-benchmark`` backend.

The benchmark has a large dependency stack that can conflict with FastVideo's
runtime. This module intentionally does not import ``av_bench``. It launches a
small runner with a caller-selected Python interpreter, so exact official
metrics can live in a separate virtual environment.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from fastvideo.logger import init_logger

logger = init_logger(__name__)


def run_av_benchmark(
    *,
    audio_dir: str | Path,
    gt_cache: str | Path,
    prediction_cache: str | Path | None = None,
    output: str | Path | None = None,
    python_executable: str | Path | None = None,
    device: str = "cuda",
    batch_size: int = 32,
    num_workers: int = 0,
    audio_length: float = 8.0,
    recompute: bool = False,
    skip_video_related: bool = False,
    skip_clap: bool = False,
    align_prediction_keys: bool = False,
) -> Path:
    """Extract prediction features and evaluate with official av-benchmark.

    ``python_executable`` should point to an isolated environment containing
    the official ``av_bench`` package. Omitting it uses the current process,
    which is useful only when that package is already installed.
    """
    audio_path = Path(audio_dir).expanduser().resolve()
    gt_path = Path(gt_cache).expanduser().resolve()
    if not audio_path.is_dir():
        raise FileNotFoundError(f"Generated audio directory does not exist: {audio_path}")
    if not gt_path.is_dir():
        raise FileNotFoundError(f"Ground-truth cache directory does not exist: {gt_path}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}")
    if audio_length <= 0:
        raise ValueError(f"audio_length must be positive, got {audio_length}")

    root = audio_path.parent
    prediction_path = (Path(prediction_cache).expanduser().resolve() if prediction_cache is not None else root /
                       "av_benchmark_cache")
    output_path = (Path(output).expanduser().resolve() if output is not None else root / "av_benchmark_results.json")
    runner = Path(__file__).with_name("_av_benchmark_runner.py")
    python = str(Path(python_executable).expanduser()) if python_executable is not None else sys.executable

    command = [
        python,
        str(runner),
        "--audio-dir",
        str(audio_path),
        "--gt-cache",
        str(gt_path),
        "--prediction-cache",
        str(prediction_path),
        "--output",
        str(output_path),
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--audio-length",
        str(audio_length),
    ]
    if recompute:
        command.append("--recompute")
    if skip_video_related:
        command.append("--skip-video-related")
    if skip_clap:
        command.append("--skip-clap")
    if align_prediction_keys:
        command.append("--align-prediction-keys")

    logger.info("Launching official av-benchmark with isolated Python: %s", python)
    logger.info("Prediction cache: %s", prediction_path)
    subprocess.run(command, check=True)
    if not output_path.is_file():
        raise RuntimeError(f"av-benchmark completed without writing expected output: {output_path}")
    return output_path
