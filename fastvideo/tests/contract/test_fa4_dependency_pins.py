# SPDX-License-Identifier: Apache-2.0
"""Keep every FA4 installation surface on the same tested revision."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _match(path: str, pattern: str) -> str:
    source = (REPO_ROOT / path).read_text(encoding="utf-8")
    match = re.search(pattern, source)
    assert match is not None, f"FA4 pin not found in {path}"
    return match.group(1)


def test_fa4_dependency_pins_match() -> None:
    pins = {
        "pyproject.toml": _match(
            "pyproject.toml",
            r'flash-attn-4 = .*rev = "([0-9a-f]{40})"'),
        "docker/Dockerfile": _match(
            "docker/Dockerfile", r"ARG FA4_CUTE_REF=([0-9a-f]{40})"),
        "apps/dreamverse/pyproject.toml": _match(
            "apps/dreamverse/pyproject.toml",
            r"flash-attention\.git@([0-9a-f]{40})"),
        "fastvideo-kernel/README.md": _match(
            "fastvideo-kernel/README.md",
            r"flash-attention\.git@([0-9a-f]{40})"),
    }

    assert len(set(pins.values())) == 1, f"FA4 dependency pins disagree: {pins}"
