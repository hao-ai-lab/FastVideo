#!/usr/bin/env python3
"""Fail scientific RVM launches when tracked source differs from the Git SHA."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


def capture(*args: str, cwd: Path) -> str:
    return subprocess.check_output(
        args,
        cwd=cwd,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    if os.environ.get("RVM_ALLOW_DIRTY_SOURCE", "0") == "1":
        print(
            "WARNING: RVM_ALLOW_DIRTY_SOURCE=1; source reproducibility gate skipped."
        )
        return
    if not (root / ".git").exists():
        raise RuntimeError(
            f"RVM scale-up requires a Git checkout, but {root} has no .git directory"
        )

    status = capture(
        "git",
        "status",
        "--porcelain",
        "--untracked-files=no",
        cwd=root,
    )
    if status:
        raise RuntimeError(
            "Tracked source differs from the checked-out commit. Commit or revert "
            f"the following changes before training:\n{status}"
        )

    manifest = {
        "head": capture("git", "rev-parse", "HEAD", cwd=root),
        "tree": capture("git", "rev-parse", "HEAD^{tree}", cwd=root),
        "branch": capture(
            "git",
            "rev-parse",
            "--abbrev-ref",
            "HEAD",
            cwd=root,
        ),
        "clean": True,
    }
    output = os.environ.get("RVM_SOURCE_MANIFEST")
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
