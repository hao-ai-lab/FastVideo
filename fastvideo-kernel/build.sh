#!/bin/bash
set -ex

# Simple build script wrapping uv/pip
# Usage:
#   ./build.sh  # local build (torch-based arch detection, TK only on SM90)
# Environment overrides (if set, they win over auto-detection):
#   TORCH_CUDA_ARCH_LIST
#   CMAKE_ARGS (for FASTVIDEO_KERNEL_BUILD_TK / CMAKE_CUDA_ARCHITECTURES / GPU_BACKEND)

echo "Building fastvideo-kernel..."

# ---------------------------------------------------------------------------
# Neutralise conda-injected compiler toolchains.
#
# Conda compiler packages (gcc_linux-aarch64, gxx_linux-64, etc.) set
# CMAKE_ARGS, CFLAGS, CXXFLAGS, and LDFLAGS on activation.  When multiple
# toolchains are installed the variables can reference a *cross*-compiler
# that doesn't match the host (e.g. aarch64-conda-linux-gnu-c++ on x86_64).
# Even when the correct toolchain is active, the flags it injects
# (-march=nocona, -mtune=haswell, …) can conflict with nvcc's host-compiler
# expectations.  Clear them so CMake discovers the system compiler instead.
# ---------------------------------------------------------------------------
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    _need_clean=0
    # Detect conda cross-compiler that doesn't match the host.
    _host_arch="$(uname -m)"
    if [[ "${CXX:-}" == *"conda"* ]] || [[ "${CC:-}" == *"conda"* ]]; then
        _need_clean=1
    fi
    if [[ "${CMAKE_ARGS:-}" == *"conda"* ]]; then
        _need_clean=1
    fi
    if (( _need_clean )); then
        echo "NOTE: Clearing conda-injected compiler settings (CC/CXX/CMAKE_ARGS/CFLAGS/...)"
        echo "      to use the system compiler for CUDA extension builds."
        unset CC CXX CMAKE_ARGS CFLAGS CXXFLAGS LDFLAGS
    fi
    unset _need_clean _host_arch
fi

# Ensure submodules are initialized if needed (tk)
git submodule update --init --recursive

# Install build dependencies
uv pip install scikit-build-core cmake ninja

RELEASE=0
GPU_BACKEND=CUDA
for arg in "$@"; do
    case "$arg" in
        --rocm)
            GPU_BACKEND=ROCM
            ;;
    esac
done

has_cmake_arg() {
    local key="$1"
    [[ "${CMAKE_ARGS:-}" =~ (^|[[:space:]])-D${key}(=|$) ]]
}

detect_with_torch() {
    # Prefer the active venv's python directly over `uv run --active --no-project`,
    # which on some uv versions provisions its own interpreter and misses packages
    # installed into VIRTUAL_ENV.
    local py
    if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
        py="${VIRTUAL_ENV}/bin/python"
    else
        py="$(command -v python3 || command -v python)"
    fi
    "${py}" -c "import torch
if not torch.cuda.is_available():
    raise RuntimeError('torch.cuda.is_available() is false')
mj, mn = torch.cuda.get_device_capability(0)
print(f'{mj}.{mn}')"
}

if [ "${GPU_BACKEND}" = "CUDA" ]; then
    detected_cc="$(detect_with_torch)" || {
        echo "ERROR: torch-based CUDA arch detection failed in uv environment." >&2
        echo "       Ensure torch is installed and CUDA is available in the uv-selected Python." >&2
        exit 1
    }

    cc_major="${detected_cc%%.*}"
    cc_minor="${detected_cc##*.}"
    cmake_arch="${cc_major}${cc_minor}"
    echo "Detected compute capability via torch: ${detected_cc} (sm_${cmake_arch})"

    # Respect explicit overrides.
    if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
        if [ "${cc_major}" = "9" ] && [ "${cc_minor}" = "0" ]; then
            export TORCH_CUDA_ARCH_LIST="9.0a"
        else
            export TORCH_CUDA_ARCH_LIST="${cc_major}.${cc_minor}"
        fi
    fi

    # ThunderKittens build targeting:
    # - SM90: compile Hopper/TK kernels with 90a.
    # - Others (e.g., SM100): compile non-TK path with detected arch.
    if ! has_cmake_arg "CMAKE_CUDA_ARCHITECTURES"; then
        if [ "${cc_major}" = "9" ] && [ "${cc_minor}" = "0" ]; then
            CMAKE_ARGS="${CMAKE_ARGS:-} -DCMAKE_CUDA_ARCHITECTURES=90a"
        else
            CMAKE_ARGS="${CMAKE_ARGS:-} -DCMAKE_CUDA_ARCHITECTURES=${cmake_arch}"
        fi
    fi

    if ! has_cmake_arg "FASTVIDEO_KERNEL_BUILD_TK"; then
        if [ "${cc_major}" = "9" ] && [ "${cc_minor}" = "0" ]; then
            CMAKE_ARGS="${CMAKE_ARGS:-} -DFASTVIDEO_KERNEL_BUILD_TK=ON"
        else
            CMAKE_ARGS="${CMAKE_ARGS:-} -DFASTVIDEO_KERNEL_BUILD_TK=OFF"
        fi
    fi
fi

if ! has_cmake_arg "GPU_BACKEND"; then
    CMAKE_ARGS="${CMAKE_ARGS:-} -DGPU_BACKEND=${GPU_BACKEND}"
fi
export CMAKE_ARGS

echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST:-<unset>}"
echo "CMAKE_ARGS: ${CMAKE_ARGS:-<unset>}"
echo "GPU_BACKEND: ${GPU_BACKEND:-<unset>}"
# Build and install
# Use -v for verbose output
uv pip install . -v --no-build-isolation
