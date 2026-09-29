# SPDX-License-Identifier: Apache-2.0
"""Real subprocess imports with CUDA hidden; no kernel or platform stubs."""

import importlib.util
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
def test_public_entrypoints_without_visible_cuda(cuda_hidden, arguments):
    if arguments[0] == "-m" and importlib.util.find_spec("fastapi") is None:
        pytest.skip("fastapi is required for CLI entrypoint smoke tests")
    completed = subprocess.run(
        [sys.executable, *arguments], capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def _write_fake_kernel(root: Path) -> None:
    kernel_package = root / "fastvideo_kernel"
    kernel_package.mkdir()
    (kernel_package / "__init__.py").write_text(
        "raise RuntimeError('kernel package initializer must not run')\n", encoding="utf-8",
    )
    (kernel_package / "ops.py").write_text("", encoding="utf-8")
    (kernel_package / "vmoba.py").write_text("", encoding="utf-8")

    flash_attn_package = root / "flash_attn"
    flash_attn_package.mkdir()
    (flash_attn_package / "__init__.py").write_text("__version__ = '2.7.4'\n", encoding="utf-8")


def test_kernel_availability_probes_find_submodules_without_importing_parent(cuda_hidden, tmp_path: Path):
    _write_fake_kernel(tmp_path)
    # Put the fake packages ahead of any installed fastvideo_kernel/flash_attn.
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; sys.path.insert(0, {str(tmp_path)!r}); "
            "from fastvideo.utils import is_vmoba_available, is_vsa_available; "
            "assert is_vsa_available() is True; "
            "assert is_vmoba_available() is True; "
            "assert 'fastvideo_kernel' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.fixture
def kernel_probe(monkeypatch, tmp_path: Path):
    """Run the probes in-process against packages under ``tmp_path`` only."""
    from fastvideo import utils

    for name in ("fastvideo_kernel", "flash_attn"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    utils.is_vsa_available.cache_clear()
    utils.is_vmoba_available.cache_clear()
    yield utils
    utils.is_vsa_available.cache_clear()
    utils.is_vmoba_available.cache_clear()


def test_kernel_probes_reject_namespace_submodules(kernel_probe, tmp_path: Path):
    # Directory submodules without __init__.py are namespace portions, whose
    # lookup needs the (deliberately unimported) parent in sys.modules.
    kernel_package = tmp_path / "fastvideo_kernel"
    kernel_package.mkdir()
    (kernel_package / "__init__.py").write_text("raise RuntimeError('must not run')\n", encoding="utf-8")
    (kernel_package / "ops").mkdir()
    (kernel_package / "vmoba").mkdir()
    assert kernel_probe.is_vsa_available() is False
    assert kernel_probe.is_vmoba_available() is False
    assert "fastvideo_kernel" not in sys.modules


def test_kernel_probes_reject_single_file_module(kernel_probe, tmp_path: Path):
    # A single-file fastvideo_kernel has no submodules. Without the guard,
    # PathFinder would search sys.path for these unrelated top-level modules.
    (tmp_path / "fastvideo_kernel.py").write_text("raise RuntimeError('must not run')\n", encoding="utf-8")
    (tmp_path / "ops.py").write_text("", encoding="utf-8")
    (tmp_path / "vmoba.py").write_text("", encoding="utf-8")
    assert kernel_probe.is_vsa_available() is False
    assert kernel_probe.is_vmoba_available() is False
    assert "fastvideo_kernel" not in sys.modules


def test_kernel_probes_find_package_submodules_in_process(kernel_probe, tmp_path: Path):
    _write_fake_kernel(tmp_path)
    assert kernel_probe.is_vsa_available() is True
    assert kernel_probe.is_vmoba_available() is True
    assert "fastvideo_kernel" not in sys.modules


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
