from __future__ import annotations

import sys
import types
from pathlib import Path

from tests._bootstrap import ensure_local_kernel_sources_first


def test_bootstrap_prefers_local_kernel_checkout(monkeypatch) -> None:
    tests_root = Path(__file__).resolve().parent
    kernel_root = tests_root.parent
    repo_root = kernel_root.parent
    kernel_python_root = kernel_root / "python"

    stale_module = types.ModuleType("fastvideo_kernel")
    stale_module.__file__ = (
        "/tmp/site-packages/fastvideo_kernel/__init__.py"
    )
    monkeypatch.setitem(sys.modules, "fastvideo_kernel", stale_module)
    monkeypatch.setattr(sys, "path", ["/tmp/site-packages"])

    ensure_local_kernel_sources_first()

    assert "fastvideo_kernel" not in sys.modules
    assert sys.path[:3] == [
        str(kernel_python_root),
        str(kernel_root),
        str(repo_root),
    ]
