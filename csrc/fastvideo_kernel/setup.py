import os
import subprocess
import sys
from pathlib import Path
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).parent.absolute()
CSRC_DIR = ROOT / "csrc"

# Path to ThunderKittens (TK)
def get_tk_dir():
    tk_env = os.getenv("THUNDERKITTENS_ROOT")
    if tk_env:
        return tk_env
    
    # Check common locations
    possible_paths = [
        ROOT / "tk",
        ROOT / "csrc" / "tk",
        ROOT.parent / "attn" / "sliding_tile_attn" / "tk",
        ROOT.parent / "attn" / "video_sparse_attn" / "tk",
    ]
    for p in possible_paths:
        if (p / "include" / "kittens.cuh").exists():
            return str(p)
    
    # Default fallback
    return str(ROOT.parent / "attn" / "sliding_tile_attn" / "tk")

TK_DIR = get_tk_dir()

def get_cuda_flags(tk_root: str) -> list:
    python_include = subprocess.check_output(
        ["python", "-c", "import sysconfig; print(sysconfig.get_path('include'))"]
    ).decode().strip()
    
    torch_includes = subprocess.check_output([
        "python", "-c",
        "import torch; from torch.utils.cpp_extension import include_paths; "
        "print(' '.join(['-I' + p for p in include_paths()]))"
    ]).decode().strip().split()
    
    return [
        "-DNDEBUG",
        "-Xcompiler=-Wno-psabi",
        "-Xcompiler=-fno-strict-aliasing",
        "--expt-extended-lambda",
        "--expt-relaxed-constexpr",
        "-forward-unknown-to-host-compiler",
        "--use_fast_math",
        "-std=c++20",
        "-O3",
        "-Xnvlink=--verbose",
        "-Xptxas=--verbose",
        "-Xptxas=--warn-on-spills",
        f"-I{tk_root}/include",
        f"-I{tk_root}/prototype",
        f"-I{python_include}",
        "-DTORCH_COMPILE",
        "-DKITTENS_HOPPER",
        "-arch=sm_90a",
    ] + torch_includes

def get_extensions():
    if not torch.cuda.is_available():
        return []

    extensions = []
    cpp_flags = ["-std=c++20", "-O3"]
    
    # Check if TK is available
    if not os.path.exists(os.path.join(TK_DIR, "include", "kittens.cuh")):
        print(f"Warning: ThunderKittens not found at {TK_DIR}. CUDA kernels will not be built.")
        return []

    cuda_flags = get_cuda_flags(TK_DIR)
    
    # STA Extension
    extensions.append(CUDAExtension(
        "fastvideo_kernel._C.st_attn",
        sources=[
            "csrc/st_attn.cpp",
            "csrc/st_attn_h100.cu",
        ],
        extra_compile_args={
            "cxx": cpp_flags + ["-DTK_COMPILE_ST_ATTN"], 
            "nvcc": cuda_flags + ["-DTK_COMPILE_ST_ATTN"]
        },
        libraries=["cuda"],
    ))
    
    # VSA Extension
    extensions.append(CUDAExtension(
        "fastvideo_kernel._C.vsa",
        sources=[
            "csrc/vsa.cpp",
            "csrc/block_sparse_h100.cu",
        ],
        extra_compile_args={
            "cxx": cpp_flags + ["-DTK_COMPILE_BLOCK_SPARSE"], 
            "nvcc": cuda_flags + ["-DTK_COMPILE_BLOCK_SPARSE"]
        },
        libraries=["cuda"],
    ))
    
    return extensions

ext_modules = []
if not any(arg in sys.argv for arg in ["clean", "egg_info", "--version"]):
    try:
        import torch
        ext_modules = get_extensions()
    except Exception as e:
        print(f"Warning: Failed to configure CUDA extensions: {e}")

setup(
    name="fastvideo-kernel",
    version="0.1.0",
    description="Unified CUDA kernels for FastVideo",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    author="Hao AI Lab",
    url="https://github.com/hao-ai-lab/FastVideo",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension} if ext_modules else {},
    python_requires=">=3.10",
    install_requires=["torch>=2.5.0", "triton>=2.0.0"],
)
