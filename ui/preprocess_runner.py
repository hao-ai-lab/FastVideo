# SPDX-License-Identifier: Apache-2.0
"""
Runs preprocessing subprocess for datasets.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("fastvideo.ui.preprocess_runner")


def build_preprocess_args(
    dataset_id: str,
    raw_path: str,
    output_dir: str,
    workload_type: str,
    model_path: str,
    dataset_type: str = "merged",
    num_gpus: int = 1,
) -> list[str]:
    """Build CLI args for v1_preprocessing_new."""
    return [
        "--model-path", model_path,
        "--mode", "preprocess",
        "--workload-type", workload_type,
        "--preprocess.video_loader_type", "torchvision",
        "--preprocess.dataset_type", dataset_type,
        "--preprocess.dataset_path", raw_path,
        "--preprocess.dataset_output_dir", output_dir,
        "--preprocess.preprocess_video_batch_size", "2",
        "--preprocess.dataloader_num_workers", "0",
        "--preprocess.max_height", "480",
        "--preprocess.max_width", "832",
        "--preprocess.num_frames", "77",
        "--preprocess.train_fps", "16",
        "--preprocess.samples_per_file", "8",
        "--preprocess.flush_frequency", "8",
        "--preprocess.video_length_tolerance_range", "5",
    ]


def run_preprocess(
    dataset_id: str,
    raw_path: str,
    output_dir: str,
    workload_type: str,
    model_path: str,
    dataset_type: str,
    num_gpus: int,
    log_file_path: str,
    on_status_change: Callable[[str, str | None], None],
    stop_event: threading.Event,
) -> None:
    """Run preprocessing subprocess. Calls on_status_change(status, error)."""
    repo_root = Path(__file__).resolve().parent.parent
    preprocess_module = "fastvideo.pipelines.preprocess.v1_preprocessing_new"

    args = build_preprocess_args(
        dataset_id=dataset_id,
        raw_path=raw_path,
        output_dir=output_dir,
        workload_type=workload_type,
        model_path=model_path,
        dataset_type=dataset_type,
        num_gpus=num_gpus,
    )

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node", str(num_gpus),
        "--nnodes", "1",
        "-m", preprocess_module,
    ] + args

    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"

    try:
        on_status_change("preprocessing", None)
        with open(log_file_path, "w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                cmd,
                cwd=str(repo_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in iter(proc.stdout.readline, ""):
                if stop_event.is_set():
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    on_status_change("stopped", "Preprocessing stopped by user")
                    return
                line = line.rstrip()
                if line:
                    log_file.write(line + "\n")
                    log_file.flush()

            proc.wait()
            exit_code = proc.returncode or 0

        if exit_code == 0:
            on_status_change("ready", None)
        else:
            on_status_change("failed", f"Preprocessing exited with code {exit_code}")
    except Exception as exc:
        on_status_change("failed", f"{type(exc).__name__}: {exc}")
        logger.exception("Preprocessing failed for dataset %s", dataset_id)
