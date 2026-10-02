"""The app depends on FastVideo and nothing else that serves models.

This app began as a port of a deployment built on the Reactor runtime, whose
serve process, RPC decorators and wire schema it no longer uses -- the model
and the broadcast run in one process now. That is easy to regress by copying
one more module across, so the contract is a test: no `reactor_*` import may
reappear, and the modules that must stay importable without a GPU must stay
importable without a GPU.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parents[1]

# Modules that must import with no torch, no fastvideo and no GPU: the config
# and queue logic is pure Python so it can be tested anywhere, and the entry
# point has to be able to print a dependency error rather than raise one.
CPU_ONLY_MODULES = (
    "infinite_livestream.clip_plan",
    "infinite_livestream.clip_queue",
    "infinite_livestream.config",
    "infinite_livestream.group_tag",
)


def _module_files() -> list[pathlib.Path]:
    return sorted(p for p in PACKAGE.rglob("*.py") if "tests" not in p.parts)


def _imported_names(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


@pytest.mark.parametrize("path", _module_files(), ids=lambda p: p.name)
def test_no_reactor_imports(path: pathlib.Path) -> None:
    offenders = {name for name in _imported_names(path) if name.split(".")[0].startswith("reactor")}
    assert not offenders, f"{path.name} imports {sorted(offenders)}"


@pytest.mark.parametrize("module", CPU_ONLY_MODULES)
def test_imports_without_a_gpu(module: str) -> None:
    """These must not drag torch or fastvideo in as a side effect.

    Run in a fresh interpreter, because "did importing X pull in torch" is a
    question about ``sys.modules``, which is global: any earlier test that
    imported fastvideo would make this pass or fail for reasons that have
    nothing to do with the module under test. A subprocess is the only honest
    way to ask it.
    """
    import subprocess
    import sys

    probe = (
        f"import {module}, sys; "
        "leaked = [m for m in ('torch', 'fastvideo') if m in sys.modules]; "
        "print(','.join(leaked))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=PACKAGE.parents[1],
    )
    assert result.returncode == 0, f"{module} failed to import:\n{result.stderr}"
    leaked = result.stdout.strip()
    assert not leaked, f"{module} imported {leaked} at module level"


def test_backend_defers_heavy_imports() -> None:
    """backend.py names fastvideo only inside functions.

    Module-level would make the config tests, and the entry point's dependency
    message, need a GPU.
    """
    tree = ast.parse((PACKAGE / "backend.py").read_text(encoding="utf-8"))
    top_level = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0
    }
    heavy = {name for name in top_level if name.split(".")[0] in {"torch", "torchaudio", "fastvideo"}}
    assert not heavy, f"backend.py imports {sorted(heavy)} at module level"
