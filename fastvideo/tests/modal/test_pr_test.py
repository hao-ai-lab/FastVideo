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
    spec = importlib.util.spec_from_file_location(
        "modal_pr_test_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_checkout_repository_retries_clone_and_fetches_exact_pr_commit(
        monkeypatch):
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
        lambda path, ignore_errors: cleanup_paths.append(
            (path, ignore_errors)),
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
        {"cwd": "/", "check": False},
        {"cwd": "/", "check": False},
        {"cwd": "/tmp/FastVideo", "check": False},
        {"cwd": "/tmp/FastVideo", "check": False},
        {"cwd": "/tmp/FastVideo", "check": False},
    ]

    fetch_command = commands[2][0]
    assert fetch_command[-2:] == [
        "origin",
        "0123456789abcdef0123456789abcdef01234567",
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
        lambda path, ignore_errors: cleanup_paths.append(
            (path, ignore_errors)),
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
        ("https://example.com/repo.git", "0123456789abcdef", "1584"),
        ("https://example.com/repo.git",
         "0123456789abcdef0123456789abcdef01234567", "not-a-pr"),
        ("https://example.com/repo.git",
         "0123456789abcdef0123456789abcdef01234567", "0"),
    ],
)
def test_checkout_repository_rejects_invalid_buildkite_values(
        monkeypatch, git_repo, git_commit, pr_number):
    module = _load_pr_test_module(monkeypatch)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "git must not run for invalid input"),
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
def test_run_test_command_composes_valid_post_checkout_shell(
        monkeypatch, build_kernel, install_command):
    module = _load_pr_test_module(monkeypatch)
    real_run = subprocess.run
    events = []

    monkeypatch.setenv(
        "BUILDKITE_REPO", "https://github.com/hao-ai-lab/FastVideo.git")
    monkeypatch.setenv(
        "BUILDKITE_COMMIT", "0123456789abcdef0123456789abcdef01234567")
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

    assert [event for event, _ in events] == ["checkout", "run"]
    shell_command = events[1][1][-1]
    assert "cd /FastVideo" in shell_command
    assert "git clone" not in shell_command
    assert ("./build.sh" in shell_command) is build_kernel
    if install_command:
        assert install_command in shell_command
    else:
        assert "uv pip install -e" not in shell_command
    real_run(["/bin/bash", "-n"], input=shell_command, text=True, check=True)
