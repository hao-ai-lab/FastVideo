#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"
source .venv/bin/activate

OUTPUT_DIR="${OUTPUT_DIR:-outputs/mmaudio_small_44k_ema_300000_vggsound_test}"
MANIFEST="${MANIFEST:-${OUTPUT_DIR}/eval_manifest.jsonl}"
REFERENCE_AUDIO_CACHE="${REFERENCE_AUDIO_CACHE:-${OUTPUT_DIR}/reference_audio}"
RESULTS="${RESULTS:-${OUTPUT_DIR}/fastvideo_eval_results.json}"
NUM_GPUS="${NUM_GPUS:-4}"
EXTRACT_WORKERS="${EXTRACT_WORKERS:-16}"
METRICS="${METRICS:-audio.frechet_distance,audio.kl_divergence,audio.clap_score,audio.desync}"

fastvideo eval run \
  --manifest "${MANIFEST}" \
  --metrics "${METRICS}" \
  --num-gpus "${NUM_GPUS}" \
  --extract-audio "${REFERENCE_AUDIO_CACHE}" \
  --extract-workers "${EXTRACT_WORKERS}" \
  --output-format full \
  --output "${RESULTS}"

echo "FastVideo evaluation results: ${RESULTS}"
