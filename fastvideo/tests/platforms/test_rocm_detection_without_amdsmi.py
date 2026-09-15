# SPDX-License-Identifier: Apache-2.0
"""ROCm detection must not depend on the amdsmi Python package.

rocm/pytorch images carry a ROCm (HIP) torch build but no amdsmi package, so
the platform has to be recognized from torch and the AMD device node alone.
"""

import sys

import pytest
import torch

import fastvideo.platforms as platforms

ROCM_PLATFORM = "fastvideo.platforms.rocm.RocmPlatform"


@pytest.fixture
def no_amdsmi(monkeypatch):
    # A None entry in sys.modules makes `import amdsmi` raise ImportError.
    monkeypatch.setitem(sys.modules, "amdsmi", None)


def test_hip_torch_with_amd_device_node_is_rocm(monkeypatch, no_amdsmi):
    monkeypatch.setattr(torch.version, "hip", "7.14.1", raising=False)
    monkeypatch.setattr(platforms, "_rocm_device_node_accessible", lambda: True)
    assert platforms.rocm_platform_plugin() == ROCM_PLATFORM


def test_hip_torch_without_amd_device_node_is_not_rocm(monkeypatch, no_amdsmi):
    monkeypatch.setattr(torch.version, "hip", "7.14.1", raising=False)
    monkeypatch.setattr(platforms, "_rocm_device_node_accessible", lambda: False)
    assert platforms.rocm_platform_plugin() is None


def test_cuda_torch_is_not_rocm_even_with_the_device_node(monkeypatch, no_amdsmi):
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(platforms, "_rocm_device_node_accessible", lambda: True)
    assert platforms.rocm_platform_plugin() is None
