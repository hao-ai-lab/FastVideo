"""Test import helpers for preferring the in-tree fastvideo-kernel sources."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _prepend_import_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _purge_package(name: str) -> None:
    prefix = f"{name}."
    for module_name in tuple(sys.modules):
        if module_name == name or module_name.startswith(prefix):
            sys.modules.pop(module_name, None)


def _module_is_from_checkout(module_file: str | None, checkout_root: Path) -> bool:
    if module_file is None:
        return False

    try:
        module_path = Path(module_file).resolve()
    except OSError:
        return False

    return module_path.is_relative_to(checkout_root.resolve())


def ensure_local_kernel_sources_first() -> None:
    tests_root = Path(__file__).resolve().parent
    kernel_root = tests_root.parent
    repo_root = kernel_root.parent
    kernel_python_root = kernel_root / "python"

    # Keep the in-tree kernel sources ahead of any preinstalled wheel so tests
    # exercise the checkout under review.
    for path in (repo_root, kernel_root, kernel_python_root):
        _prepend_import_path(path)

    importlib.invalidate_caches()

    loaded_kernel = sys.modules.get("fastvideo_kernel")
    if loaded_kernel is None:
        return

    if not _module_is_from_checkout(
        getattr(loaded_kernel, "__file__", None),
        kernel_python_root,
    ):
        _purge_package("fastvideo_kernel")
