# SPDX-License-Identifier: Apache-2.0
"""ROCm detection must not depend on the amdsmi Python package.

rocm/pytorch images carry a ROCm (HIP) torch build but no amdsmi package, so
the platform has to be recognized from torch and the AMD device nodes alone.
"""

import os
import sys
import types

import pytest
import torch

import fastvideo.platforms as platforms
from fastvideo.platforms.rocm import RocmPlatform

ROCM_PLATFORM = "fastvideo.platforms.rocm.RocmPlatform"
READ_WRITE = os.R_OK | os.W_OK


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


def test_current_platform_resolves_to_rocm_without_amdsmi(monkeypatch, no_amdsmi):
    monkeypatch.setattr(platforms, "mps_platform_plugin", lambda: None)
    monkeypatch.setattr(torch.version, "hip", "7.14.1", raising=False)
    monkeypatch.setattr(platforms, "_rocm_device_node_accessible", lambda: True)
    # Drop any platform resolved earlier in the session so the lazy
    # `current_platform` lookup runs the resolver again.
    monkeypatch.setattr(platforms, "_current_platform", None)
    monkeypatch.setattr(platforms, "_init_trace", "")
    assert platforms.resolve_current_platform_cls_qualname() == ROCM_PLATFORM
    assert isinstance(platforms.current_platform, RocmPlatform)


def fake_device_nodes(monkeypatch, accessible, render_nodes):
    """Stub the device files: `accessible` paths pass os.access, `render_nodes` is what the glob finds."""
    seen = []

    def fake_access(path, mode):
        seen.append((path, mode))
        return path in accessible

    def fake_glob(pattern):
        assert pattern == "/dev/dri/renderD*"  # codespell:ignore renderd
        return list(render_nodes)

    monkeypatch.setattr(platforms.os, "access", fake_access)
    monkeypatch.setattr(platforms.glob, "glob", fake_glob)
    return seen


def test_device_node_probe_needs_the_compute_node_and_a_render_node(monkeypatch):
    seen = fake_device_nodes(monkeypatch, {"/dev/kfd", "/dev/dri/renderD128"}, ["/dev/dri/renderD128"])
    assert platforms._rocm_device_node_accessible() is True
    assert seen == [("/dev/kfd", READ_WRITE), ("/dev/dri/renderD128", READ_WRITE)]


def test_device_node_probe_rejects_a_container_without_a_render_node(monkeypatch):
    fake_device_nodes(monkeypatch, {"/dev/kfd"}, [])
    assert platforms._rocm_device_node_accessible() is False


def test_device_node_probe_rejects_render_nodes_it_cannot_open(monkeypatch):
    fake_device_nodes(monkeypatch, {"/dev/kfd"}, ["/dev/dri/renderD128", "/dev/dri/renderD129"])
    assert platforms._rocm_device_node_accessible() is False


def test_device_node_probe_stops_without_the_compute_node(monkeypatch):
    seen = fake_device_nodes(monkeypatch, {"/dev/dri/renderD128"}, ["/dev/dri/renderD128"])
    assert platforms._rocm_device_node_accessible() is False
    assert seen == [("/dev/kfd", READ_WRITE)]


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
