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

# Ensure only the kernel's required headers are initialized. A repository-wide
# update also clones the unrelated VBench evaluation submodule. Skip outside a
# git checkout (e.g. Docker contexts that exclude .git), where the submodule
# contents must already be present.
if git rev-parse --git-dir >/dev/null 2>&1; then
    git submodule update --init --recursive include/cutlass include/tk
fi
# Fail fast with a clear message if the headers are still missing (e.g. a
# Docker context that excluded .git AND the submodule contents) instead of
# dying later in a wall of nvcc include errors. CUTLASS is consumed by the
# always-built turbodiffusion sources, so it is a hard error; ThunderKittens
# only feeds the TK-gated Hopper kernels, so a missing tree just warns (the
# TK gate resolves later, and non-SM90/ROCm builds never touch it).
if [ ! -d include/cutlass/include ]; then
    echo "ERROR: include/cutlass/include is missing. Outside a git checkout the" >&2
    echo "       CUTLASS sources must already be present (run" >&2
    echo "       'git submodule update --init --recursive include/cutlass include/tk'" >&2
    echo "       in the source checkout, or include them in the build context)." >&2
    exit 1
fi
if [ ! -d include/tk/include ]; then
    echo "WARNING: include/tk/include is missing; ThunderKittens (Hopper sm_90a)" >&2
    echo "         kernels cannot be built. Fine for non-SM90/ROCm targets." >&2
fi

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
    # Compute capability drives the arch/TK defaults below. Prefer an explicit
    # TORCH_CUDA_ARCH_LIST (works on GPU-less build machines such as CI/Docker);
    # only probe a live GPU via torch when no arch was provided.
    if [ -n "${TORCH_CUDA_ARCH_LIST:-}" ]; then
        echo "Using TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} (skipping torch GPU probe)"
        first_arch="${TORCH_CUDA_ARCH_LIST%%[;, ]*}"   # first entry, e.g. 9.0a
        first_arch="${first_arch%[af]}"                # strip trailing a/f suffix
        cc_major="${first_arch%%.*}"
        cc_minor="${first_arch##*.}"
    else
        detected_cc="$(detect_with_torch)" || {
            echo "ERROR: torch-based CUDA arch detection failed and TORCH_CUDA_ARCH_LIST is unset." >&2
            echo "       Set TORCH_CUDA_ARCH_LIST (e.g. 9.0a) for GPU-less builds, or build where CUDA is available." >&2
            exit 1
        }
        cc_major="${detected_cc%%.*}"
        cc_minor="${detected_cc##*.}"
        echo "Detected compute capability via torch: ${detected_cc}"
    fi
    cmake_arch="${cc_major}${cc_minor}"

    # Respect explicit overrides.
    if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
        if [ "${cc_major}" = "9" ] && [ "${cc_minor}" = "0" ]; then
            export TORCH_CUDA_ARCH_LIST="9.0a"
        elif [ "${cc_major}" = "12" ] && [ "${cc_minor}" = "0" ]; then
            # Blackwell sm_120 needs the arch-conditional 'a' suffix so CMake's
            # AUTO gate (matches 12.0a/120a/sm_120a) builds the attn_qat_infer
            # (modified SageAttention3 FP4) kernels instead of silently skipping.
            export TORCH_CUDA_ARCH_LIST="12.0a"
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
