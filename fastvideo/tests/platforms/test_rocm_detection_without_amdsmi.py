# SPDX-License-Identifier: Apache-2.0
"""ROCm detection must not depend on the amdsmi Python package.

rocm/pytorch images carry a ROCm (HIP) torch build but no amdsmi package, so
the platform has to be recognized from torch and the AMD device node alone.
"""

import os
import sys
import types

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


def test_device_node_probe_checks_the_amd_compute_node(monkeypatch):
    seen = []

    def fake_access(path, mode):
        seen.append((path, mode))
        return True

    monkeypatch.setattr(platforms.os, "access", fake_access)
    assert platforms._rocm_device_node_accessible() is True
    assert seen == [("/dev/kfd", os.R_OK | os.W_OK)]


def test_amdsmi_device_wins_over_the_torch_fallback(monkeypatch):
    fake_amdsmi = types.ModuleType("amdsmi")
    fake_amdsmi.amdsmi_init = lambda: None
    fake_amdsmi.amdsmi_get_processor_handles = lambda: [object()]
    fake_amdsmi.amdsmi_shut_down = lambda: None
    monkeypatch.setitem(sys.modules, "amdsmi", fake_amdsmi)

    def unexpected_probe():
        raise AssertionError("the torch fallback must not run when amdsmi finds a device")

    monkeypatch.setattr(platforms, "_rocm_device_node_accessible", unexpected_probe)
    assert platforms.rocm_platform_plugin() == ROCM_PLATFORM
