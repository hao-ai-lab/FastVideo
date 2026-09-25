# SPDX-License-Identifier: Apache-2.0
"""Optional fastvideo_kernel must not be imported at module import time."""

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

_IMPORT_SCRIPT = """
import sys

# Simulate a host without the compiled kernel package.
sys.modules["fastvideo_kernel"] = None

from fastvideo.pipelines.stages import denoising
from fastvideo.attention.backends import video_sparse_attn, vmoba

assert "fastvideo_kernel" not in sys.modules or sys.modules["fastvideo_kernel"] is None
print("ok")
"""


def test_pipeline_and_sparse_backends_import_without_kernel_package():
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    completed = subprocess.run(
        [sys.executable, "-c", _IMPORT_SCRIPT],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(_REPO_ROOT),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "ok" in completed.stdout
