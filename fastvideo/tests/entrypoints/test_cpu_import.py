# SPDX-License-Identifier: Apache-2.0
"""Real subprocess imports with CUDA hidden; no kernel or platform stubs."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("arguments", [
    ["-c", "from fastvideo import VideoGenerator, PipelineConfig, SamplingParam; "
     "import sys, torch; assert not torch.cuda.is_available(); "
     "PipelineConfig(); SamplingParam(); "
     "from fastvideo.utils import is_vsa_available; is_vsa_available(); "
     "assert 'fastvideo_kernel' not in sys.modules"],
    ["-m", "fastvideo.entrypoints.cli.main", "--help"],
    ["-m", "fastvideo.entrypoints.cli.main", "generate", "--help"],
])
def test_public_entrypoints_without_visible_cuda(arguments):
    if arguments[0] == "-m" and importlib.util.find_spec("fastapi") is None:
        pytest.skip("fastapi is required for CLI entrypoint smoke tests")
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    completed = subprocess.run(
        [sys.executable, *arguments], env=environment, capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_kernel_availability_probes_find_submodules_without_importing_parent(tmp_path: Path):
    kernel_package = tmp_path / "fastvideo_kernel"
    kernel_package.mkdir()
    (kernel_package / "__init__.py").write_text(
        "raise RuntimeError('kernel package initializer must not run')\n", encoding="utf-8",
    )
    (kernel_package / "ops.py").write_text("", encoding="utf-8")
    (kernel_package / "vmoba.py").write_text("", encoding="utf-8")

    flash_attn_package = tmp_path / "flash_attn"
    flash_attn_package.mkdir()
    (flash_attn_package / "__init__.py").write_text("__version__ = '2.7.4'\n", encoding="utf-8")

    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), environment.get("PYTHONPATH", "")],
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "from fastvideo.utils import is_vmoba_available, is_vsa_available; "
            "assert is_vsa_available() is True; "
            "assert is_vmoba_available() is True; "
            "assert 'fastvideo_kernel' not in sys.modules",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("exception_name", ["ModuleNotFoundError", "RuntimeError"])
def test_vmoba_kernel_errors_propagate_at_use(monkeypatch, exception_name):
    import builtins

    from fastvideo.attention.backends.vmoba import VMOBAAttentionImpl

    real_import = builtins.__import__
    error_type = getattr(builtins, exception_name)

    def fail_kernel_import(name, *args, **kwargs):
        if name == "fastvideo_kernel":
            raise error_type("kernel import failure sentinel")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_kernel_import)
    impl = object.__new__(VMOBAAttentionImpl)
    with pytest.raises(error_type, match="kernel import failure sentinel"):
        impl.forward(None, None, None, None)
