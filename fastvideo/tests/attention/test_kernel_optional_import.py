# SPDX-License-Identifier: Apache-2.0
"""Optional fastvideo_kernel must not be imported at module import time."""

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]

_IMPORT_SCRIPT = """
import sys

{setup}

from fastvideo.pipelines.stages import denoising
from fastvideo.attention.backends import video_sparse_attn, vmoba

assert sys.modules.get("fastvideo_kernel") is None
print("ok")
"""

# Absent: the import raises ImportError, which a module-level try/except
# around the import would also tolerate. Broken: initialization raises
# RuntimeError (for example, no visible GPU driver), which only a deferred
# import tolerates, so this case covers both the VSA and VMOBA deferrals.
_KERNEL_SETUPS = {
    "absent": 'sys.modules["fastvideo_kernel"] = None',
    "broken_init": "sys.path.insert(0, {root!r})",
}


@pytest.mark.parametrize("kernel", sorted(_KERNEL_SETUPS))
def test_pipeline_and_sparse_backends_import_without_kernel_package(cuda_hidden, tmp_path, kernel):
    kernel_package = tmp_path / "fastvideo_kernel"
    kernel_package.mkdir()
    (kernel_package / "__init__.py").write_text(
        "raise RuntimeError('kernel package initializer must not run')\n", encoding="utf-8",
    )
    setup = _KERNEL_SETUPS[kernel].format(root=str(tmp_path))
    completed = subprocess.run(
        [sys.executable, "-c", _IMPORT_SCRIPT.format(setup=setup)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(_REPO_ROOT),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "ok" in completed.stdout
