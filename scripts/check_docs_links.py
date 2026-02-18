#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate local markdown links under docs/."""

from __future__ import annotations

import re
from pathlib import Path

LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def is_external(target: str) -> bool:
    return target.startswith(
        ("http://", "https://", "mailto:", "#", "<", "javascript:"))


def resolve_candidates(file_path: Path, docs_root: Path, target: str) -> list[Path]:
    clean_target = target.split("#", 1)[0].split("?", 1)[0].strip()
    if not clean_target:
        return []

    if clean_target.endswith("/"):
        clean_target = f"{clean_target}index.md"

    candidates: list[Path] = []
    if clean_target.startswith("/"):
        candidates.append(docs_root / clean_target.lstrip("/"))
        return candidates

    base = file_path.parent
    path_target = Path(clean_target)
    candidates.append(base / path_target)

    if not path_target.suffix:
        candidates.append(base / f"{clean_target}.md")

    return candidates


def check_links(docs_root: Path) -> list[str]:
    errors: list[str] = []
    for md_file in sorted(docs_root.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        for line_no, line in enumerate(content.splitlines(), start=1):
            for target in LINK_PATTERN.findall(line):
                stripped = target.strip()
                if is_external(stripped):
                    continue

                candidates = resolve_candidates(md_file, docs_root, stripped)
                if not candidates:
                    continue

                if any(path.is_file() for path in candidates):
                    continue

                errors.append(
                    f"{md_file.relative_to(docs_root.parent)}:{line_no} -> {stripped}"
                )

    return errors


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    docs_root = repo_root / "docs"
    errors = check_links(docs_root)

    if not errors:
        print("Docs link check passed.")
        return 0

    print("Docs link check failed:")
    for err in errors:
        print(f"  - {err}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
