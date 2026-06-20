#!/usr/bin/env bash
# Profile a Wan2.1 T2V 1.3B inference benchmark with Nsight Systems.
#
# Intended environment:
#   - FastVideo Python 3.10-3.12 environment
#   - CUDA-visible NVIDIA GPU(s)
#   - Nsight Systems CLI available on PATH
#
# Usage:
#   bash scripts/performance/profile_wan_t2v_1_3b_nsys.sh
#
# Optional overrides:
#   PYTHON_BIN=python3.12
#   NSYS_BIN=nsys
#   CONFIG_PATH=.buildkite/performance-benchmarks/tests/wan-t2v-1.3b.json
#   OUTPUT_ROOT=outputs/nsys
#   NSYS_EXTRA_ARGS="--some-nsys-flag=value"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
NSYS_BIN="${NSYS_BIN:-nsys}"
CONFIG_PATH="${CONFIG_PATH:-.buildkite/performance-benchmarks/tests/wan-t2v-1.3b.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/nsys}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import sys

if not ((3, 10) <= sys.version_info[:2] <= (3, 12)):
    raise SystemExit(f"Expected Python 3.10-3.12, got {sys.version.split()[0]}")
PY

if ! command -v "${NSYS_BIN}" >/dev/null 2>&1; then
  echo "Nsight Systems CLI not found: ${NSYS_BIN}. Install Nsight Systems before profiling." >&2
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Benchmark config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

BENCHMARK_ID="$("${PYTHON_BIN}" - <<'PY' "${CONFIG_PATH}"
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    print(json.load(f)["benchmark_id"])
PY
)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${OUTPUT_ROOT}/${BENCHMARK_ID}/${TIMESTAMP}"

mkdir -p "${RUN_DIR}"

NSYS_ARGS=(
  profile
  --force-overwrite=true
  --capture-range=cudaProfilerApi
  --capture-range-end=stop
  --sample=none
  --cpuctxsw=none
  --stats=true
  --output "${RUN_DIR}/${BENCHMARK_ID}"
)

NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,cudnn,cublas,osrt}"

if [[ "${NSYS_ENABLE_NCCL_TRACE:-0}" == "1" ]]; then
  if "${NSYS_BIN}" profile --help 2>&1 | grep -Eq "(^|[[:space:],'])nccl([[:space:],']|$)"; then
    NSYS_TRACE="${NSYS_TRACE},nccl"
  else
    echo "Skipping NCCL trace: ${NSYS_BIN} does not list nccl as a supported trace domain."
  fi
fi

NSYS_ARGS+=(--trace="${NSYS_TRACE}")

if "${NSYS_BIN}" profile --help 2>&1 | grep -q -- "--cuda-trace-scope"; then
  NSYS_ARGS+=(--cuda-trace-scope=process-tree)
fi

if "${NSYS_BIN}" profile --help 2>&1 | grep -q -- "--trace-fork-before-exec"; then
  NSYS_ARGS+=(--trace-fork-before-exec=true)
fi

if [[ -n "${NSYS_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=(${NSYS_EXTRA_ARGS})
  NSYS_ARGS+=("${EXTRA_ARGS[@]}")
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export FASTVIDEO_STAGE_LOGGING="${FASTVIDEO_STAGE_LOGGING:-1}"

echo "Repo root: ${REPO_ROOT}"
echo "Python: $("${PYTHON_BIN}" --version)"
echo "Nsight Systems: $("${NSYS_BIN}" --version | head -n 1)"
echo "Config: ${CONFIG_PATH}"
echo "Output: ${RUN_DIR}"
"${PYTHON_BIN}" - <<'PY'
try:
    import torch
except ImportError:
    print("Torch: unavailable")
else:
    print(f"Torch: {torch.__version__}")
    print(f"Torch CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for idx in range(torch.cuda.device_count()):
            print(
                f"CUDA device {idx}: {torch.cuda.get_device_name(idx)} "
                f"capability={torch.cuda.get_device_capability(idx)}"
            )
PY
echo "Starting nsys profile..."

"${NSYS_BIN}" "${NSYS_ARGS[@]}" \
  "${PYTHON_BIN}" scripts/performance/run_inference_profile_from_config.py \
    --config "${CONFIG_PATH}" \
    --output-dir "${RUN_DIR}/generated_videos" \
    --summary-path "${RUN_DIR}/profile_summary.json"

echo "Done. Profile outputs are under ${RUN_DIR}"
