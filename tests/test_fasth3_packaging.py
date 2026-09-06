# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version


REPO_ROOT = Path(__file__).resolve().parents[1]


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

    assert 'TORCH_CUDA_ARCH_LIST="9.0a;10.0a;12.0a"' in workflow
    assert 'TORCH_CUDA_ARCH_LIST="10.0a;12.0a"' in workflow
    assert "arch=compute_100a,code=sm_100a" in cmake
    assert "arch=compute_103a,code=sm_103a" in cmake
    assert "patchelf==0.17.2.4" in workflow


def test_sm103a_gencode_requires_cuda_12_9():
    cmake = (REPO_ROOT / "fastvideo-kernel" / "CMakeLists.txt").read_text(encoding="utf-8")
    extension = (REPO_ROOT / "fastvideo-kernel" / "csrc" / "common_extension.cpp").read_text(encoding="utf-8")
    backend = (REPO_ROOT / "fastvideo-kernel" / "python" / "fastvideo_kernel" /
               "block_sparse_attn_sm100a.py").read_text(encoding="utf-8")
    vsa_options = cmake.index("set(_VSA_COMPILE_OPTIONS")
    sm103_guard = cmake.index("if(NOT CUDAToolkit_VERSION VERSION_LESS 12.9)", vsa_options)
    sm103_gencode = cmake.index("arch=compute_103a,code=sm_103a", sm103_guard)

    assert sm103_guard < sm103_gencode < cmake.index("endif()", sm103_guard)
    assert "list(APPEND COMPILE_DEFS TK_COMPILE_BLOCK_SPARSE_VSA_SM103A)" in cmake
    assert 'm.attr("_has_vsa_sm103a") = true' in extension
    assert 'getattr(_C, "_has_vsa_sm103a", False)' in backend
