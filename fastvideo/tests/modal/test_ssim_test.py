# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


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


class _FakeApp:

    def function(self, *_args, **_kwargs):
        def decorator(func):
            return func

        return decorator

    def local_entrypoint(self, *_args, **_kwargs):
        def decorator(func):
            return func

        return decorator


def _load_ssim_test_module(monkeypatch):
    fake_modal = types.SimpleNamespace(
        App=lambda: _FakeApp(),
        Image=_FakeImage,
        Volume=_FakeVolume,
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    module_path = Path(__file__).with_name("ssim_test.py")
    spec = importlib.util.spec_from_file_location("modal_ssim_test_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_spawn_ssim_task_passes_hf_api_key_to_pytest_env(monkeypatch, tmp_path):
    module = _load_ssim_test_module(monkeypatch)
    captured = {}

    class FakeProcess:
        pass

    def fake_popen(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)

    task = module.SSIMTask(
        task_id=0,
        test_file="fastvideo/tests/ssim/test_example.py",
        required_gpus=1,
        model_id="model-a",
    )

    running_task = module._spawn_ssim_task(
        task=task,
        repo_root=str(tmp_path),
        assigned_gpu_ids=["2"],
        log_dir=str(tmp_path),
        task_index=0,
        pytest_extra_args=["--ssim-bootstrap-mode"],
        hf_api_key="hf_test_token",
    )

    env = captured["kwargs"]["env"]
    assert env["HF_API_KEY"] == "hf_test_token"
    assert env["HF_HOME"] == "/root/data/.cache"
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["FASTVIDEO_SSIM_MODEL_ID"] == "model-a"
    assert running_task.process.__class__ is FakeProcess
    running_task.log_handle.close()


def test_prepare_workspace_uses_nonshared_kernel_install(monkeypatch):
    module = _load_ssim_test_module(monkeypatch)
    captured = {}
    task = module.SSIMTask(
        task_id=0,
        test_file="fastvideo/tests/ssim/test_example.py",
        required_gpus=1,
    )

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "_discover_ssim_tasks", lambda *_args, **_kwargs: [task])

    repo_root, tasks = module._prepare_ssim_workspace(
        git_repo="https://example.com/FastVideo.git",
        git_commit="0123456789abcdef",
        pr_number="false",
        hf_api_key="hf_test_token",
    )

    setup_command = captured["args"][2]
    assert "kernel_build_cache.py install" in setup_command
    assert "--cache-root" not in setup_command
    assert repo_root == "/FastVideo"
    assert tasks == [task]
    assert module.SSIM_COMMON_KWARGS["volumes"] == {"/root/data": module.model_vol}
    assert not hasattr(module, "kernel_cache_vol")
