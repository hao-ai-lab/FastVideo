import os
import shutil
import subprocess
from pathlib import Path

import pytest


BUILD_SCRIPT = Path(__file__).parents[1] / "build.sh"


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(0o755)


def _run_build(tmp_path: Path, capability: str, cuda_release: str) -> tuple[subprocess.CompletedProcess[str], Path]:
    kernel_dir = tmp_path / "fastvideo-kernel"
    kernel_dir.mkdir()
    shutil.copy2(BUILD_SCRIPT, kernel_dir / "build.sh")
    (kernel_dir / "include/cutlass/include").mkdir(parents=True)
    (kernel_dir / "include/tk/include").mkdir(parents=True)

    result_dir = tmp_path / "result"
    result_dir.mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "git", "#!/bin/sh\nexit 1\n")
    _write_executable(fake_bin / "python3", "#!/bin/sh\nprintf '%s\\n' \"$FAKE_CUDA_CAPABILITY\"\n")
    _write_executable(
        fake_bin / "nvcc",
        "#!/bin/sh\nprintf 'Cuda compilation tools, release %s, V%s.0\\n' \"$FAKE_CUDA_RELEASE\" \"$FAKE_CUDA_RELEASE\"\n",
    )
    _write_executable(
        fake_bin / "uv",
        "#!/bin/sh\nprintf '%s' \"${TORCH_CUDA_ARCH_LIST:-}\" > \"$FV_TEST_RESULT/arch\"\n"
        "printf '%s' \"${CMAKE_ARGS:-}\" > \"$FV_TEST_RESULT/cmake_args\"\n",
    )

    env = os.environ.copy()
    for name in ("CMAKE_ARGS", "CONDA_PREFIX", "CUDA_HOME", "CUDACXX", "TORCH_CUDA_ARCH_LIST", "VIRTUAL_ENV"):
        env.pop(name, None)
    env.update({
        "FAKE_CUDA_CAPABILITY": capability,
        "FAKE_CUDA_RELEASE": cuda_release,
        "FV_TEST_RESULT": str(result_dir),
        "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
    })
    result = subprocess.run(
        ["bash", "build.sh"],
        cwd=kernel_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, result_dir


@pytest.mark.parametrize(
    "capability,torch_arch,cmake_arch",
    [("9.0", "9.0a", "90a"), ("12.0", "12.0a", "120"), ("12.1", "12.1a", "121")],
)
def test_build_script_maps_architecture_specific_targets(
    tmp_path: Path, capability: str, torch_arch: str, cmake_arch: str
) -> None:
    result, result_dir = _run_build(tmp_path, capability, "13.0")

    assert result.returncode == 0, result.stderr
    assert (result_dir / "arch").read_text() == torch_arch
    assert f"-DCMAKE_CUDA_ARCHITECTURES={cmake_arch}" in (result_dir / "cmake_args").read_text()


def test_build_script_rejects_sm121_before_cmake_on_cuda_12(tmp_path: Path) -> None:
    result, _ = _run_build(tmp_path, "12.1", "12.9")

    assert result.returncode != 0
    assert "sm_121a requires CUDA Toolkit 13.0+" in result.stderr
