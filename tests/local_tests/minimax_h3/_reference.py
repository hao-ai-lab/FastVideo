# SPDX-License-Identifier: Apache-2.0

import hashlib
import inspect
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_ROOT = Path(
    os.environ.get("MINIMAX_H3_OFFICIAL_REF_DIR") or REPO_ROOT / "DiffusersMiniMaxH3"
).resolve()
REFERENCE_SRC = REFERENCE_ROOT / "src"
PINNED_COMMIT = "abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc"


def assert_pinned_reference(relative_path: str, sha256: str) -> Path:
    path = REFERENCE_ROOT / relative_path
    if not path.is_file():
        pytest.skip(
            f"pinned MiniMax-H3 Diffusers reference is missing: {path}",
            allow_module_level=True,
        )
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != sha256:
        raise RuntimeError(
            f"MiniMax-H3 reference file changed from {PINNED_COMMIT}: {relative_path} "
            f"has SHA-256 {actual_sha256}, expected {sha256}."
        )
    return path


def assert_reference_source(symbol: object, relative_path: str) -> None:
    expected = (REFERENCE_ROOT / relative_path).resolve()
    source = inspect.getsourcefile(symbol)
    actual = Path(source).resolve() if source is not None else None
    if actual != expected:
        raise RuntimeError(
            f"MiniMax-H3 parity imported {actual}, expected {expected}. "
            f"Prepend {REFERENCE_SRC} to PYTHONPATH before starting pytest."
        )
