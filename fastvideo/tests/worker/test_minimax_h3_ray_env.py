# SPDX-License-Identifier: Apache-2.0
"""CPU checks for settings applied before H3 Ray actor imports."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

import fastvideo.worker.minimax_h3_disaggregated as runtime_module


@pytest.fixture(autouse=True)
def clean_fa4_environment(monkeypatch):
    monkeypatch.delenv("FASTVIDEO_FA4", raising=False)
    monkeypatch.setattr(runtime_module, "RAY_NON_CARRY_OVER_ENV_VARS", set())


@pytest.mark.parametrize("setting", ["0", "1"])
def test_actor_env_carries_explicit_driver_fa4_without_node_identity(monkeypatch, setting):
    monkeypatch.setenv("FASTVIDEO_FA4", setting)
    monkeypatch.setenv("FASTVIDEO_HOST_IP", "192.168.23.1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "driver-nic")
    args = SimpleNamespace(ray_runtime_env=None)
    assert runtime_module._h3_actor_runtime_env(args) == {"env_vars": {"FASTVIDEO_FA4": setting}}


def test_unset_driver_flag_preserves_worker_defaults():
    assert runtime_module._h3_actor_runtime_env(SimpleNamespace(ray_runtime_env=None)) == {}


def test_configured_actor_environment_wins_without_mutating_input(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_FA4", "1")
    configured = {"env_vars": {"FASTVIDEO_FA4": "0", "EXISTING": "keep"}, "working_dir": "/tmp/project"}
    original = deepcopy(configured)
    actual = runtime_module._h3_actor_runtime_env(SimpleNamespace(ray_runtime_env=configured))
    assert actual == {"env_vars": original["env_vars"]}
    actual["env_vars"]["EXISTING"] = "changed"
    assert configured == original


def test_driver_flag_merges_with_existing_environment(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_FA4", "1")
    configured = {"env_vars": {"EXISTING": "keep"}, "pip": ["example-package"]}
    actual = runtime_module._h3_actor_runtime_env(SimpleNamespace(ray_runtime_env=configured))
    assert actual == {"env_vars": {"EXISTING": "keep", "FASTVIDEO_FA4": "1"}}
    assert "FASTVIDEO_FA4" not in configured["env_vars"]


def test_actor_environment_inherits_job_packages_instead_of_reusing_local_paths(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_FA4", "1")
    configured = {"working_dir": "/tmp/project", "py_modules": ["/tmp/local-module"]}
    assert runtime_module._h3_actor_runtime_env(SimpleNamespace(ray_runtime_env=configured)) == {
        "env_vars": {"FASTVIDEO_FA4": "1"}
    }


def test_driver_flag_respects_non_carry_over_policy(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_FA4", "1")
    monkeypatch.setattr(runtime_module, "RAY_NON_CARRY_OVER_ENV_VARS", {"FASTVIDEO_FA4"})
    assert runtime_module._h3_actor_runtime_env(SimpleNamespace(ray_runtime_env=None)) == {}


@pytest.mark.parametrize("initialized", [False, True])
def test_both_actor_creation_options_receive_fa4_even_with_existing_ray(monkeypatch, initialized):
    monkeypatch.setenv("FASTVIDEO_FA4", "1")
    options = []
    init_calls = []

    class ActorBuilder:
        def options(self, **kwargs):
            options.append(kwargs)
            return self

        def remote(self, *args):
            return object()

    fake_ray = SimpleNamespace(
        is_initialized=lambda: initialized,
        init=lambda **kwargs: init_calls.append(kwargs),
        cluster_resources=lambda: {"node:192.168.23.1": 1.0, "node:192.168.23.2": 1.0},
        remote=lambda cls: ActorBuilder(),
    )
    monkeypatch.setattr(runtime_module, "ray", fake_ray)
    monkeypatch.setattr(runtime_module, "assert_ray_available", lambda: None)
    monkeypatch.setattr(runtime_module.RayMiniMaxH3DisaggregatedRuntime, "_validate_workers", lambda self: None)
    configured = {"env_vars": {"EXISTING": "keep"}}
    runtime_module.RayMiniMaxH3DisaggregatedRuntime(
        SimpleNamespace(ray_runtime_env=configured),
        encoder_node_ip="192.168.23.1",
        dit_node_ip="192.168.23.2",
    )
    assert len(init_calls) == (0 if initialized else 1)
    assert len(options) == 2
    for actor_options in options:
        assert actor_options["runtime_env"] == {"env_vars": {"EXISTING": "keep", "FASTVIDEO_FA4": "1"}}
    assert configured == {"env_vars": {"EXISTING": "keep"}}


def test_worker_diagnostic_reports_actual_python_and_fa4(monkeypatch):
    messages = []
    monkeypatch.setenv("FASTVIDEO_FA4", "1")
    monkeypatch.setattr(runtime_module, "get_ip", lambda: "192.168.23.2")
    monkeypatch.setattr(runtime_module.logger, "info", lambda fmt, *args: messages.append(fmt % args))
    runtime_module._log_h3_worker_environment("dit")
    line, = messages
    assert "role=dit node_ip=192.168.23.2" in line
    assert f"python={runtime_module.sys.executable}" in line
    assert "FASTVIDEO_FA4=1" in line
