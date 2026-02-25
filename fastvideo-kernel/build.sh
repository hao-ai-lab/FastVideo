#!/bin/bash
set -ex

# Simple build script wrapping uv/pip
# Usage:
#   ./build.sh  # local build (torch-based arch detection, TK only on SM90)
# Environment overrides (if set, they win over auto-detection):
#   TORCH_CUDA_ARCH_LIST
#   CMAKE_ARGS (for FASTVIDEO_KERNEL_BUILD_TK / CMAKE_CUDA_ARCHITECTURES / GPU_BACKEND)

echo "Building fastvideo-kernel..."

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
    uv run --active --no-project python -c "import torch
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
