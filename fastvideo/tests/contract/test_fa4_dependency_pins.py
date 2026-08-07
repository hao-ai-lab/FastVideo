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


def test_fa4_blackwell_installs_select_cuda13_runtime() -> None:
    root_pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    dreamverse_pyproject = (REPO_ROOT / "apps/dreamverse/pyproject.toml").read_text(encoding="utf-8")
    dockerfile = (REPO_ROOT / "docker/Dockerfile").read_text(encoding="utf-8")
    dreamverse_dockerfile = (REPO_ROOT / "apps/dreamverse/docker/Dockerfile").read_text(encoding="utf-8")
    kernel_readme = (REPO_ROOT / "fastvideo-kernel/README.md").read_text(encoding="utf-8")

    assert '"flash-attn-4",' in root_pyproject
    assert '"flash-attn-4 @ git+' in dreamverse_pyproject
    assert 'if [ "${UV_TORCH_BACKEND}" = "cu130" ]' in dockerfile
    assert 'FA4_PACKAGE="flash-attn-4[cu13]"' in dockerfile
    assert 'FA4_PACKAGE="flash-attn-4"' in dockerfile
    assert 'elif [ "${TARGETARCH}" = "arm64" ]' in dockerfile
    assert '"${UV_TORCH_BACKEND}" == "cu130"' in dreamverse_dockerfile
    assert '"nvidia-cutlass-dsl[cu13]"' in dreamverse_dockerfile
    assert "flash-attn-4[cu13]" in kernel_readme
