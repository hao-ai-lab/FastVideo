# SPDX-License-Identifier: Apache-2.0
"""Import hygiene for the lightweight Apple Silicon MLX test path."""

from __future__ import annotations

import subprocess
import sys


def test_mlx_runtime_import_does_not_eager_import_video_generator() -> None:
    code = (
        "import sys; "
        "import fastvideo.mlx_runtime.memory; "
        "print('fastvideo.entrypoints.video_generator' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "False"
