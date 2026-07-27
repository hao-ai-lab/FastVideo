#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/mmaudio_small_44k_ema_300000_vggsound_test}"
AUDIO_DIR="${AUDIO_DIR:-${OUTPUT_DIR}/audio}"
GT_CACHE="${GT_CACHE:-/mnt/lustre/vlm-kai/datasets/VGGSound/av_benchmark/vggsound-test-eval-cache}"
PREDICTION_CACHE="${PREDICTION_CACHE:-${OUTPUT_DIR}/av_benchmark_cache}"
RESULTS="${RESULTS:-${OUTPUT_DIR}/av_benchmark_results.json}"
AV_BENCH_PYTHON="${AV_BENCH_PYTHON:-/mnt/lustre/vlm-kai/av-benchmark/.venv/bin/python}"
BATCH_SIZE="${BATCH_SIZE:-32}"
# The official extractor initializes CUDA models before constructing its
# DataLoader. Zero avoids CUDA-after-fork hangs and this container's 64 MB shm.
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-cuda}"
RECOMPUTE="${RECOMPUTE:-0}"
SKIP_VIDEO_RELATED="${SKIP_VIDEO_RELATED:-0}"
# The official VGGSound GT cache has no CLAP text features, so CLAP is not an
# official VGGSound output metric. Skip its prediction passes by default.
SKIP_CLAP="${SKIP_CLAP:-1}"
# Official VGGSound precomputed cache sanitizes a subset of sample ids by
# prefixing underscores. Align only unique matches inside prediction caches.
ALIGN_PREDICTION_KEYS="${ALIGN_PREDICTION_KEYS:-1}"

if [[ ! -x "${AV_BENCH_PYTHON}" ]]; then
  echo "Missing isolated av-benchmark Python: ${AV_BENCH_PYTHON}" >&2
  echo "Set AV_BENCH_PYTHON=/path/to/av-benchmark/.venv/bin/python" >&2
  exit 2
fi
if [[ ! -d "${GT_CACHE}" ]]; then
  echo "Missing official VGGSound GT cache: ${GT_CACHE}" >&2
  echo "Set GT_CACHE=/path/to/vggsound-test-eval-cache" >&2
  exit 2
fi

EXTRA_ARGS=()
if [[ "${RECOMPUTE}" == "1" ]]; then
  EXTRA_ARGS+=(--recompute)
fi
if [[ "${SKIP_VIDEO_RELATED}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-video-related)
fi
if [[ "${SKIP_CLAP}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-clap)
fi
if [[ "${ALIGN_PREDICTION_KEYS}" == "1" ]]; then
  EXTRA_ARGS+=(--align-prediction-keys)
fi

.venv/bin/fastvideo eval v2a \
  --backend av-benchmark \
  --audio-dir "${AUDIO_DIR}" \
  --gt-cache "${GT_CACHE}" \
  --prediction-cache "${PREDICTION_CACHE}" \
  --output "${RESULTS}" \
  --python-executable "${AV_BENCH_PYTHON}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --audio-length 8 \
  "${EXTRA_ARGS[@]}"

echo "Official av-benchmark results: ${RESULTS}"
