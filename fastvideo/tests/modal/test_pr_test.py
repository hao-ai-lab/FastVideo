# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest


class _FakeImage:

    @classmethod
    def from_registry(cls, *_args, **_kwargs):
        return cls()

    def apt_install(self, *_args, **_kwargs):
        return self

    def run_commands(self, *_args, **_kwargs):
        return self

    def env(self, *_args, **_kwargs):
        return self


class _FakeVolume:

    @classmethod
    def from_name(cls, *_args, **_kwargs):
        return cls()


class _FakeSecret:

    @classmethod
    def from_dict(cls, *_args, **_kwargs):
        return cls()


class _FakeApp:

    def function(self, *_args, **_kwargs):

        def decorator(func):
            return func

        return decorator


def _load_pr_test_module(monkeypatch):
    fake_modal = types.SimpleNamespace(
        App=lambda: _FakeApp(),
        Image=_FakeImage,
        Secret=_FakeSecret,
        Volume=_FakeVolume,
    )
    fake_image_utils = types.SimpleNamespace(
        resolve_image_ref=lambda image_ref: image_ref,
        resolve_uv_torch_backend=lambda _image_tag: None,
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setitem(sys.modules, "modal_image_utils", fake_image_utils)
    module_path = Path(__file__).with_name("pr_test.py")
    spec = importlib.util.spec_from_file_location("modal_pr_test_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_checkout_repository_retries_clone_and_fetches_pr_ref(monkeypatch):
    module = _load_pr_test_module(monkeypatch)
    commands = []
    cleanup_paths = []
    sleep_seconds = []
    returncodes = iter([128, 0, 0, 0, 0])

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        return types.SimpleNamespace(returncode=next(returncodes))

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        module.shutil,
        "rmtree",
        lambda path, ignore_errors: cleanup_paths.append((path, ignore_errors)),
    )
    monkeypatch.setattr(module.time, "sleep", sleep_seconds.append)

    module._checkout_repository(
        "https://github.com/hao-ai-lab/FastVideo.git",
        "0123456789abcdef0123456789abcdef01234567",
        "1584",
        repo_root="/tmp/FastVideo",
    )

    assert len(commands) == 5
    clone_command = commands[0][0]
    assert commands[1][0] == clone_command
    assert clone_command == [
        "git",
        "-c",
        "http.version=HTTP/1.1",
        "clone",
        "--config",
        "http.version=HTTP/1.1",
        "--depth=1",
        "--filter=blob:none",
        "--no-checkout",
        "https://github.com/hao-ai-lab/FastVideo.git",
        "/tmp/FastVideo",
    ]
    assert cleanup_paths == [
        ("/tmp/FastVideo", True),
        ("/tmp/FastVideo", True),
    ]
    assert sleep_seconds == [5]
    assert [kwargs for _, kwargs in commands] == [
        {
            "cwd": "/",
            "check": False
        },
        {
            "cwd": "/",
            "check": False
        },
        {
            "cwd": "/tmp/FastVideo",
            "check": False
        },
        {
            "cwd": "/tmp/FastVideo",
            "check": False
        },
        {
            "cwd": "/tmp/FastVideo",
            "check": False
        },
    ]

    fetch_command = commands[2][0]
    assert fetch_command[-2:] == [
        "origin",
        "refs/pull/1584/head",
    ]
    assert "--depth=1" in fetch_command
    assert "--filter=blob:none" in fetch_command
    assert commands[3][0][-3:] == ["checkout", "--detach", "FETCH_HEAD"]
    assert commands[4][0][-4:] == [
        "submodule",
        "update",
        "--init",
        "--recursive",
    ]
    assert all(command[0:3] == [
        "git",
        "-c",
        "http.version=HTTP/1.1",
    ] for command, _ in commands)


def test_checkout_repository_fetches_direct_commit(monkeypatch):
    module = _load_pr_test_module(monkeypatch)
    commands = []
    commit = "0123456789abcdef0123456789abcdef01234567"

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module.shutil, "rmtree", lambda *_args, **_kwargs: None)

    module._checkout_repository(
        "git@github.com:macthecadillac/FastVideo.git",
        commit,
        "false",
        repo_root="/tmp/FastVideo",
    )

    assert len(commands) == 4
    assert commands[1][0][-2:] == ["origin", commit]
    assert commands[2][0][-3:] == ["checkout", "--detach", "FETCH_HEAD"]


def test_git_retry_exhaustion_is_bounded_and_cleans_each_attempt(monkeypatch):
    module = _load_pr_test_module(monkeypatch)
    calls = []
    cleanup_paths = []
    sleep_seconds = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return types.SimpleNamespace(returncode=128)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        module.shutil,
        "rmtree",
        lambda path, ignore_errors: cleanup_paths.append((path, ignore_errors)),
    )
    monkeypatch.setattr(module.time, "sleep", sleep_seconds.append)

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        module._run_git_with_retries(
            ["git", "clone", "repo", "/tmp/FastVideo"],
            cwd="/",
            cleanup_path="/tmp/FastVideo",
        )

    assert len(calls) == 3
    assert cleanup_paths == [("/tmp/FastVideo", True)] * 3
    assert sleep_seconds == [5, 10]


@pytest.mark.parametrize(
    ("git_repo", "git_commit", "pr_number"),
    [
        ("", "0123456789abcdef", "false"),
        ("-bad-option", "0123456789abcdef", "false"),
        ("https://example.com/repo.git", "", "false"),
        ("https://example.com/repo.git", "not-a-commit", "false"),
        ("https://example.com/repo.git", "0123456789abcdef", "not-a-pr"),
        ("https://example.com/repo.git", "0123456789abcdef", "0"),
    ],
)
def test_checkout_repository_rejects_invalid_buildkite_values(monkeypatch, git_repo, git_commit, pr_number):
    module = _load_pr_test_module(monkeypatch)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("git must not run for invalid input"),
    )

    with pytest.raises(RuntimeError):
        module._checkout_repository(git_repo, git_commit, pr_number)


@pytest.mark.parametrize(
    ("build_kernel", "install_command"),
    [
        (True, 'uv pip install -e ".[test]"'),
        (False, ""),
    ],
)
def test_run_test_command_composes_valid_post_checkout_shell(monkeypatch, build_kernel, install_command):
    module = _load_pr_test_module(monkeypatch)
    real_run = subprocess.run
    events = []

    monkeypatch.setenv("BUILDKITE_REPO", "https://github.com/hao-ai-lab/FastVideo.git")
    monkeypatch.setenv("BUILDKITE_COMMIT", "0123456789abcdef0123456789abcdef01234567")
    monkeypatch.setenv("BUILDKITE_PULL_REQUEST", "false")
    monkeypatch.setattr(
        module,
        "_checkout_repository",
        lambda *args: events.append(("checkout", args)),
    )

    def fake_run(args, **_kwargs):
        events.append(("run", args))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module.run_test_command(
        "pytest fastvideo/tests/api -q",
        build_kernel=build_kernel,
        install_command=install_command,
    )

    assert [event for event, _ in events] == ["checkout", "run", "run"]
    setup_command = events[1][1][-1]
    test_command = events[2][1][-1]
    assert "cd /FastVideo" in setup_command
    assert "git clone" not in setup_command
    assert ("kernel_build_cache.py install" in setup_command) is build_kernel
    assert "pytest fastvideo/tests/api -q" not in setup_command
    assert "kernel_build_cache.py install" not in test_command
    assert "pytest fastvideo/tests/api -q" in test_command
    if install_command:
        assert install_command in setup_command
    else:
        assert "uv pip install -e" not in setup_command
    for shell_command in (setup_command, test_command):
        real_run(["/bin/bash", "-n"], input=shell_command, text=True, check=True)


def test_run_test_command_uses_nonshared_kernel_install_before_tests(monkeypatch):
    module = _load_pr_test_module(monkeypatch)
    commands = []

    def fake_run(args, **_kwargs):
        commands.append(args[-1])
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setenv("BUILDKITE_REPO", "https://example.com/FastVideo.git")
    monkeypatch.setenv("BUILDKITE_COMMIT", "0123456789abcdef")
    monkeypatch.setenv("BUILDKITE_PULL_REQUEST", "false")
    monkeypatch.setattr(module, "_checkout_repository", lambda *_args: None)
    monkeypatch.setattr(subprocess, "run", fake_run)

    module.run_test_command("pytest fastvideo/tests/api -q", build_kernel=True)

    assert len(commands) == 2
    setup_command, test_command = commands
    assert "kernel_build_cache.py install" in setup_command
    assert "--cache-root" not in setup_command
    assert "pytest fastvideo/tests/api -q" not in setup_command
    assert "kernel_build_cache.py install" not in test_command
    assert "pytest fastvideo/tests/api -q" in test_command
    assert not hasattr(module, "kernel_cache_vol")


def test_run_unit_test_uses_shared_command(monkeypatch):
    module = _load_pr_test_module(monkeypatch)
    commands = []
    monkeypatch.setattr(module, "run_test", commands.append)

    module.run_unit_test()

    assert commands == ["bash .buildkite/scripts/unit_test.sh"]
    unit_command = (Path(__file__).resolve().parents[3] / ".buildkite/scripts/unit_test.sh").read_text()
    for test_path in (
            "./fastvideo/tests/modal/test_kernel_build_cache.py",
            "./fastvideo/tests/modal/test_pr_test.py",
            "./fastvideo/tests/modal/test_ssim_test.py",
    ):
        assert test_path in unit_command


def test_wave1_lane_functions_use_shared_scripts(monkeypatch):
    """Modal and the self-hosted CI runner must execute the same per-lane scripts so
    their test selections cannot drift (same contract as the unit lane)."""
    module = _load_pr_test_module(monkeypatch)
    commands = []
    monkeypatch.setattr(module, "run_test", commands.append)

    hf_prefix = "export HF_HOME='/root/data/.cache' && hf auth login --token $HF_API_KEY && "
    module.run_kernel_tests()
    module.run_inference_tests_vmoba()
    module.run_golden_gate_tests()
    module.run_encoder_tests()
    module.run_vae_tests()
    module.run_transformer_tests()
    module.run_inference_lora_tests()
    module.run_distill_dmd_tests()
    module.run_train_framework_tests()

    assert commands == [
        "bash .buildkite/scripts/lanes/kernel_tests.sh",
        "bash .buildkite/scripts/lanes/inference_vmoba.sh",
        hf_prefix + "bash .buildkite/scripts/lanes/golden_gate.sh",
        hf_prefix + "bash .buildkite/scripts/lanes/encoder.sh",
        hf_prefix + "bash .buildkite/scripts/lanes/vae.sh",
        hf_prefix + "FASTVIDEO_FA4=0 bash .buildkite/scripts/lanes/transformer.sh",
        "bash .buildkite/scripts/lanes/inference_lora.sh",
        "FASTVIDEO_FA4=0 bash .buildkite/scripts/lanes/distillation_dmd.sh",
        hf_prefix + "FASTVIDEO_FA4=0 bash .buildkite/scripts/lanes/train_framework.sh",
    ]

    lanes_dir = Path(__file__).resolve().parents[3] / ".buildkite/scripts/lanes"
    for script, payload in {
            "kernel_tests.sh": "pytest fastvideo-kernel/tests/ -vs",
            "inference_vmoba.sh": "python fastvideo/tests/inference/vmoba/test_vmoba_inference.py",
            "golden_gate.sh": 'exec pytest "$golden_root" -vs',
            "encoder.sh": "pytest ./fastvideo/tests/encoders -vs",
            "vae.sh": "pytest ./fastvideo/tests/vaes -vs",
            "transformer.sh": "pytest ./fastvideo/tests/transformers -vs",
            "inference_lora.sh": "pytest ./fastvideo/tests/inference/lora/test_lora_inference_similarity.py -vs",
            "distillation_dmd.sh": "pytest ./fastvideo/tests/training/distill/test_distill_dmd.py -vs",
            "train_framework.sh": "pytest ./fastvideo/tests/train/models ./fastvideo/tests/train/methods -vs",
            "eval.sh": "pytest ./fastvideo/tests/eval -vs",
    }.items():
        assert payload in (lanes_dir / script).read_text(), script
    assert "golden_root=./fastvideo/tests/golden_gate" in (lanes_dir / "golden_gate.sh").read_text()

    # run_eval_tests goes through run_test_command (custom install extras);
    # pin its shared script + install command textually.
    eval_source = (Path(__file__).resolve().parent / "pr_test.py").read_text()
    assert "bash .buildkite/scripts/lanes/eval.sh" in eval_source
    assert 'install_command=\'uv pip install -e ".[test,eval-full]"\'' in eval_source
