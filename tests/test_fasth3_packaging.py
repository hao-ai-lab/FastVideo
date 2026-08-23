# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_fasth3_extra_and_root_kernel_pin_match_source_release():
    root_project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    kernel_project = tomllib.loads((REPO_ROOT / "fastvideo-kernel" / "pyproject.toml").read_text(encoding="utf-8"))
    kernel_version = kernel_project["project"]["version"]

    dependencies = root_project["project"]["dependencies"]
    assert f"fastvideo-kernel=={kernel_version}; sys_platform == 'linux'" in dependencies
    assert root_project["project"]["optional-dependencies"]["fasth3"] == ["flash-attn-4"]


def test_kernel_release_matrix_can_publish_sm100a_wheels():
    workflow = (REPO_ROOT / ".github" / "workflows" / "publish-kernel.yml").read_text(encoding="utf-8")

    assert 'TORCH_CUDA_ARCH_LIST="9.0a;10.0a;12.0a"' in workflow
    assert 'TORCH_CUDA_ARCH_LIST="10.0a;12.0a"' in workflow
    assert "patchelf==0.17.2.4" in workflow
