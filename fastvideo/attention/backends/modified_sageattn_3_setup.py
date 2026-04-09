import warnings
import os
from pathlib import Path
import subprocess
from typing import Any

from packaging.version import parse, Version
from setuptools import find_packages, setup
from setuptools._distutils.cmd import Command
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

this_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = Path(this_dir).resolve()
package_root = repo_dir.parents[2] / "fastvideo-kernel"

PACKAGE_NAME = "modified_sageattn"

# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("FAHOPPER_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("FAHOPPER_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("FAHOPPER_FORCE_CXX11_ABI", "FALSE") == "TRUE"


def get_cuda_bare_metal_version(cuda_dir: str) -> tuple[str, Version]:
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc.",
        stacklevel=2,
    )


def append_nvcc_threads(nvcc_extra_args: list[str]) -> list[str]:
    return nvcc_extra_args + ["--threads", "4"]


ext_modules: list[Any] = []

if not SKIP_CUDA_BUILD:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    check_if_cuda_home_none("--fahopper")
    cc_flag = []
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version < Version("12.8"):
        raise RuntimeError("Sage3 is only supported on CUDA 12.8 and above")
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_120a,code=sm_120a")

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True
    modified_sageattn_dir = (package_root / "modified_sageattn")
    cutlass_dir = repo_dir / "csrc" / "cutlass"
    (repo_dir / "csrc").mkdir(parents=True, exist_ok=True)
    if not cutlass_dir.exists():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/NVIDIA/cutlass.git",
                        str(cutlass_dir)],
                       check=True)
    nvcc_flags = [
        "-O3",
        # "-O0",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        # "--ptxas-options=-v",  # printing out number of registers
        "--ptxas-options=--verbose,--warn-on-local-memory-usage",  # printing out number of registers
        "-lineinfo",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",  # Can toggle for debugging
        "-DNDEBUG",  # Important, otherwise performance is severely impacted
        "-DQBLKSIZE=128",
        "-DKBLKSIZE=128",
        "-DCTA256",
        "-DDQINRMEM",
    ]
    include_dirs = [
        modified_sageattn_dir,
        cutlass_dir / "include",
        cutlass_dir / "tools" / "util" / "include",
    ]

    ext_modules.append(
        CUDAExtension(
            name="fp4attn_cuda",
            sources=[str(modified_sageattn_dir / "blackwell/api.cu")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": append_nvcc_threads(nvcc_flags + ["-DEXECMODE=0"] + cc_flag),
            },
            include_dirs=include_dirs,
            # Without this we get and error about cuTensorMapEncodeTiled not defined
            libraries=["cuda"]))
    ext_modules.append(
        CUDAExtension(
            name="fp4quant_cuda",
            sources=[str(modified_sageattn_dir / "quantization/fp4_quantization_4d.cu")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": append_nvcc_threads(nvcc_flags + ["-DEXECMODE=0"] + cc_flag),
            },
            include_dirs=include_dirs,
            # Without this we get and error about cuTensorMapEncodeTiled not defined
            libraries=["cuda"]))


class CachedWheelsCommand(_bdist_wheel):

    def run(self):
        super().run()


setup_cmdclass: dict[str, type[Command]]
if ext_modules:
    setup_cmdclass = {
        "bdist_wheel": CachedWheelsCommand,
        "build_ext": BuildExtension,
    }
else:
    setup_cmdclass = {
        "bdist_wheel": CachedWheelsCommand,
    }

setup(
    name=PACKAGE_NAME,
    version="1.0.0",
    packages=find_packages(where=str(package_root), exclude=(
        "build",
        "csrc",
        "tests",
        "dist",
        "docs",
        "benchmarks",
    )),
    package_dir={"": str(package_root)},
    description="FP4FlashAttention",
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass=setup_cmdclass,  # type: ignore[arg-type]
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "einops",
        "packaging",
        "ninja",
    ],
)
