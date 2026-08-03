# SPDX-License-Identifier: Apache-2.0

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_ROOT = REPO_ROOT / "DiffusersMiniMaxH3"
REFERENCE_SRC = REFERENCE_ROOT / "src"
PINNED_COMMIT = "abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc"


def assert_pinned_reference(relative_path: str, sha256: str) -> Path:
    path = REFERENCE_ROOT / relative_path
    if not path.is_file():
        raise RuntimeError(f"Pinned MiniMax-H3 Diffusers reference is missing: {path}")
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != sha256:
        raise RuntimeError(
            f"MiniMax-H3 reference file changed from {PINNED_COMMIT}: {relative_path} "
            f"has SHA-256 {actual_sha256}, expected {sha256}."
        )
    return path
