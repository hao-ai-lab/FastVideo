#!/usr/bin/env bash
set -euo pipefail

HARNESS=${HARNESS:-/mnt/benchmark_ltx2_bf16_cutlass_gemm_b2_swizzle4_20c36.py}
PYTHON=${PYTHON:-/mnt/FastVideo/.venv/bin/python}
FASTVIDEO_DIR=${FASTVIDEO_DIR:-/mnt/FastVideo}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/fastvideo_runs/pr1630_ltx2_b2_cutlass_swizzle4_$(date -u +%Y%m%dT%H%M%SZ)}
SEARCH_SPACE=${SEARCH_SPACE:-DEFAULT}
CUTLASS_DIR=${TORCHINDUCTOR_CUTLASS_DIR:-}
expected_harness_hash=4f937bc1ebc80dd2c07dadbecec84c7c7eb2d7da59aaedf52aad36f95e025add
expected_cutlass_head=e67e63c331d6e4b729047c95cf6b92c8454cba89

if [[ -z "${CUTLASS_DIR}" ]]; then
  for candidate in \
    /mnt/FastVideo/fastvideo-kernel/include/cutlass \
    /mnt/cutlass \
    /mnt/flash-attention-82d6441eec5d4dfec120153db2c0145ae855a083/csrc/cutlass \
    /mnt/flash-attention/csrc/cutlass \
    /workspace/cutlass \
    /opt/cutlass \
    /usr/local/src/cutlass; do
    if [[ -d "${candidate}/python/cutlass_library" ]]; then
      CUTLASS_DIR=${candidate}
      break
    fi
  done
fi
if [[ ! -d "${CUTLASS_DIR}/python/cutlass_library" ]]; then
  echo "Set TORCHINDUCTOR_CUTLASS_DIR to a full NVIDIA CUTLASS checkout (repo root with python/cutlass_library)." >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Python is not executable: ${PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${HARNESS}" ]]; then
  echo "Harness not found: ${HARNESS}" >&2
  exit 2
fi
test "$(sha256sum "${HARNESS}" | awk '{print $1}')" = "${expected_harness_hash}"
test "$(git -C "${CUTLASS_DIR}" rev-parse HEAD)" = "${expected_cutlass_head}"

mkdir -p "${OUTPUT_DIR}"
FASTVIDEO_COMMIT=$(git -C "${FASTVIDEO_DIR}" rev-parse HEAD)
export FASTVIDEO_COMMIT
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE=${SEARCH_SPACE}
export TORCHINDUCTOR_CUTLASS_DIR=${CUTLASS_DIR}
export TORCHINDUCTOR_CUTLASS_INSTANTIATION_LEVEL=${TORCHINDUCTOR_CUTLASS_INSTANTIATION_LEVEL:-0}
export CUTLASS_EPILOGUE_FUSION=${CUTLASS_EPILOGUE_FUSION:-0}

# Six exact SM100 configurations selected by the B1 projection gate. This B2
# gate deliberately admits no neighboring epilogue variants or swizzles.
allowlist='^cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_(bf16_bf16_256x256x64_(0x0x1_0_tnt|2x2x1_0_tnt)_align8_2sm|void_bf16_256x256x64_(2x2x1_0_ttn_align8_stream_k_2sm|0x0x1_0_ntn_align8_2sm|4x1x1_0_(ttn|ntt)_align8_2sm))$'

run_variant() {
  local variant=$1
  local tag=$2
  local autotune=$3
  local backends=$4
  local regex=${5:-}
  local swizzles=${6:-}
  TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/pr1630_cutlass_${tag}_$$" \
  TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=${autotune} \
  TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=${backends} \
  TORCHINDUCTOR_CUTLASS_ALLOWLIST=${regex} \
  CUTLASS_SWIZZLES=${swizzles} \
    "${PYTHON}" "${HARNESS}" \
      --variant "${variant}" \
      --output "${OUTPUT_DIR}/${tag}.jsonl" \
      --batch-factor 2 \
      2>&1 | tee "${OUTPUT_DIR}/${tag}.log"
}

# A/X/B controls expose clock or thermal drift.  All compilation/autotuning is
# cold in distinct cache directories and occurs before any timed sample.
run_variant current current_a 0 ATEN,TRITON,CPP
run_variant cutlass cutlass 1 ATEN,CUTLASS "$allowlist" 4
run_variant current current_b 0 ATEN,TRITON,CPP

if grep -Eiq 'illegal memory access|failed to execute choice|traceback|runtimeerror|cuda error' "${OUTPUT_DIR}/cutlass.log"; then
  echo "CUTLASS safety gate failed; refusing comparison" >&2
  exit 3
fi

"${PYTHON}" "${HARNESS}" \
  --compare \
    "${OUTPUT_DIR}/current_a.jsonl" \
    "${OUTPUT_DIR}/cutlass.jsonl" \
    "${OUTPUT_DIR}/current_b.jsonl" \
  --output "${OUTPUT_DIR}/comparison.jsonl" \
  2>&1 | tee "${OUTPUT_DIR}/comparison.log"

echo "Results: ${OUTPUT_DIR}"
