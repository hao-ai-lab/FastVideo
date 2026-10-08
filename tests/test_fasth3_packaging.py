# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version


REPO_ROOT = Path(__file__).resolve().parents[1]


def _torch_cuda_arch_lists(workflow: str) -> list[set[str]]:
    """Return the arch set of every TORCH_CUDA_ARCH_LIST build leg in a workflow."""
    return [set(value.split(";")) for value in re.findall(r'TORCH_CUDA_ARCH_LIST="([^"]+)"', workflow)]


def test_fasth3_extra_and_root_kernel_pin_match_source_release():
    root_project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    kernel_project = tomllib.loads((REPO_ROOT / "fastvideo-kernel" / "pyproject.toml").read_text(encoding="utf-8"))
    kernel_version = kernel_project["project"]["version"]

    dependencies = root_project["project"]["dependencies"]
    kernel_requirement = next(Requirement(value) for value in dependencies if value.startswith("fastvideo-kernel"))
    assert Version(kernel_version) in kernel_requirement.specifier
    fasth3_extra = root_project["project"]["optional-dependencies"]["fasth3"]
    assert "flash-attn-4" in fasth3_extra
    assert any(value.startswith("fastvideo-kernel") for value in fasth3_extra)
    kernel_sources = root_project["tool"]["uv"]["sources"]["fastvideo-kernel"]
    assert {
        "path": "fastvideo-kernel",
        "marker": "platform_machine == 'x86_64'",
        "extra": "fasth3",
    } in kernel_sources


def test_kernel_release_matrix_can_publish_data_center_blackwell_wheels():
    workflow = (REPO_ROOT / ".github" / "workflows" / "publish-kernel.yml").read_text(encoding="utf-8")
    cmake = (REPO_ROOT / "fastvideo-kernel" / "CMakeLists.txt").read_text(encoding="utf-8")

    # Check the required archs per wheel instead of exact strings: adding an
    # unrelated arch (e.g. sm_121a for DGX Spark) must not break this contract.
    arch_lists = _torch_cuda_arch_lists(workflow)
    # x86_64 cu130: Hopper TK + data-center Blackwell VSA + consumer Blackwell FP4.
    assert any({"9.0a", "10.0a", "10.3a", "12.0a"} <= archs for archs in arch_lists), arch_lists
    # aarch64 cu130: data-center + consumer Blackwell, no Hopper TK.
    assert any({"10.0a", "10.3a", "12.0a"} <= archs and "9.0a" not in archs for archs in arch_lists), arch_lists
    assert "arch=compute_100a,code=sm_100a" in cmake
    assert "arch=compute_103a,code=sm_103a" in cmake
    assert "patchelf==0.17.2.4" in workflow
