#!/bin/bash
# Build the experimental H3 SwiGLU/NVFP4 GEMM epilogue for sm_121a and install the
# shared object beside fastvideo_kernel.h3_swiglu_gemm_binding. swiglu_gemm._backend()
# falls back to that binding, so this is the in-tree way to enable try_swiglu_gemm.
#
# IMPORTANT: do NOT compile with --use_fast_math / -ftz=true. The epilogue depends on
# bf16 subnormals; FTZ flushes them and breaks parity with the Triton reference
# (test_minimax_h3_swiglu_gemm.py fails 5/13 cases at magnitude 8). Keep -ftz=false.
#
# usage: bash fastvideo-kernel/build_h3_swiglu.sh [PYTHON]
#   PYTHON must be the interpreter of the env whose fastvideo_kernel package should
#   receive the .so (default: `python` on PATH).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${1:-python}"
SITE="$("$PYTHON" -c 'import pathlib, fastvideo_kernel; print(pathlib.Path(fastvideo_kernel.__file__).parent)')"
NVCC="${CUDACXX:-${CUDA_HOME:-/usr/local/cuda}/bin/nvcc}"
ARCH="${TORCH_CUDA_ARCH_LIST:-12.1a}"
case "$ARCH" in
    12.1a|12.1) GENCODE=(-gencode "arch=compute_121a,code=sm_121a") ;;
    *) echo "This kernel targets GB10 (sm_121a); TORCH_CUDA_ARCH_LIST=$ARCH" >&2; exit 2 ;;
esac

echo "building minimax_h3_swiglu_gemm.cu for ${ARCH} -> ${SITE}"
"$NVCC" -shared -o "$SITE/lib_fastvideo_h3_swiglu_gemm.so" \
    "$ROOT/csrc/quantization/minimax_h3_swiglu_gemm.cu" \
    -I"$ROOT/include/cutlass/include" \
    -I"$ROOT/include/cutlass/tools/util/include" \
    -I"$ROOT/csrc/quantization" \
    "${GENCODE[@]}" \
    -std=c++17 -O3 -ftz=false \
    --expt-relaxed-constexpr --expt-extended-lambda \
    -DNDEBUG -Xcompiler -fPIC
echo "installed $SITE/lib_fastvideo_h3_swiglu_gemm.so"
"$PYTHON" -c "import torch, fastvideo_kernel.h3_swiglu_gemm_binding as b; print('binding:', b.__file__)"
