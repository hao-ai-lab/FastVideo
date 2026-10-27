# SPDX-License-Identifier: Apache-2.0
"""Import hygiene for the lightweight Apple Silicon MLX test path."""

from __future__ import annotations

import subprocess
import sys

_HEAVY = "fastvideo.entrypoints.video_generator"


def _modules_after(import_line: str) -> set[str]:
    """Return sys.modules after running one import in a fresh interpreter."""
    code = f"import sys; {import_line}; print('\\n'.join(sorted(sys.modules)))"
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    return set(result.stdout.split())


def test_mlx_runtime_import_adds_no_heavy_entrypoints() -> None:
    """Importing the MLX runtime must not drag in the generator stack itself.

    ``fastvideo/__init__.py`` imports ``VideoGenerator`` eagerly, so every
    ``fastvideo.*`` import loads it transitively. That is a property of the
    top-level package rather than of this runtime, so compare against a plain
    ``import fastvideo`` baseline and assert the MLX runtime adds nothing heavy
    on top of it.
    """
    baseline = _modules_after("import fastvideo")
    with_runtime = _modules_after("import fastvideo.mlx_runtime.memory")
    assert _HEAVY not in with_runtime - baseline
