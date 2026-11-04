import os
import shutil
import subprocess
from pathlib import Path

import pytest


BUILD_SCRIPT = Path(__file__).parents[1] / "build.sh"


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(0o755)


def _run_build(
    tmp_path: Path,
    capability: str,
    cuda_release: str,
    cmake_args: str | None = None,
    extra_args: list[str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path]:
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
    if cmake_args is not None:
        env["CMAKE_ARGS"] = cmake_args
    result = subprocess.run(
        ["bash", "build.sh", *(extra_args or [])],
        cwd=kernel_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, result_dir


@pytest.mark.parametrize(
    "capability,torch_arch,cmake_arch",
    [
        ("9.0", "9.0a", "90a"),
        ("10.0", "10.0a", "100"),
        ("10.3", "10.3a", "103"),
        ("12.0", "12.0a", "120"),
        ("12.1", "12.1a", "121"),
    ],
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


@pytest.mark.parametrize("capability", ["9.0", "10.3"])
def test_build_script_defers_tk_detection_to_cmake(tmp_path: Path, capability: str) -> None:
    result, result_dir = _run_build(tmp_path, capability, "13.0")

    assert result.returncode == 0, result.stderr
    assert "-DFASTVIDEO_KERNEL_BUILD_TK=AUTO" in (result_dir / "cmake_args").read_text()


@pytest.mark.parametrize("option", ["ON", "OFF"])
def test_build_script_preserves_explicit_tk_option(tmp_path: Path, option: str) -> None:
    result, result_dir = _run_build(tmp_path, "9.0", "13.0", f"-DFASTVIDEO_KERNEL_BUILD_TK={option}")

    assert result.returncode == 0, result.stderr
    assert (result_dir / "cmake_args").read_text().count("-DFASTVIDEO_KERNEL_BUILD_TK=") == 1
    assert f"-DFASTVIDEO_KERNEL_BUILD_TK={option}" in (result_dir / "cmake_args").read_text()


@pytest.mark.parametrize("cmake_args", [None, "-DOTHER_OPTION=ON", "-DFASTVIDEO_KERNEL_BUILD_EXTENSION=OFF"])
@pytest.mark.parametrize("build_args", [["--python-only"], ["--rocm", "--python-only"]])
def test_python_only_build_defaults_extension_to_off(
    tmp_path: Path, cmake_args: str | None, build_args: list[str]
) -> None:
    result, result_dir = _run_build(tmp_path, "9.0", "13.0", cmake_args, build_args)

    assert result.returncode == 0, result.stderr
    assert (result_dir / "cmake_args").read_text().count("-DFASTVIDEO_KERNEL_BUILD_EXTENSION=OFF") == 1
    if cmake_args == "-DOTHER_OPTION=ON":
        assert cmake_args in (result_dir / "cmake_args").read_text()
    if "--rocm" in build_args:
        assert "-DGPU_BACKEND=ROCM" in (result_dir / "cmake_args").read_text()


def test_python_only_build_rejects_extension_on(tmp_path: Path) -> None:
    result, _ = _run_build(tmp_path, "9.0", "13.0", "-DFASTVIDEO_KERNEL_BUILD_EXTENSION=ON", ["--python-only"])

    assert result.returncode != 0
    assert "--python-only conflicts" in result.stderr
