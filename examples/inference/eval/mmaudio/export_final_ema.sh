#!/usr/bin/env bash
set -euo pipefail

# One-time export of the synthesized PostHocEMA transformer plus the frozen
# official conditioning/decoder components into a complete FastVideo model.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"

EMA_CHECKPOINT="${EMA_CHECKPOINT:-outputs/mmaudio_small_44k_ddp_from_scratch/posthoc_ema/official_ddp/mmaudio_ema_final_sigma_0p05_step_000300000.pth}"
OUTPUT_MODEL="${OUTPUT_MODEL:-converted_weights/mmaudio/small_44k_ema_300000}"
ASSET_ROOT="${ASSET_ROOT:-official_weights/mmaudio}"

if [[ -e "${OUTPUT_MODEL}/model_index.json" ]]; then
  echo "Model already exported: ${OUTPUT_MODEL}"
  exit 0
fi

source .venv/bin/activate
python scripts/checkpoint_conversion/convert_mmaudio_to_diffusers.py \
  --variant small_44k \
  --transformer-checkpoint "${EMA_CHECKPOINT}" \
  --audio-vae-checkpoint "${ASSET_ROOT}/raw/ext_weights/v1-44.pth" \
  --synchformer-checkpoint "${ASSET_ROOT}/raw/ext_weights/synchformer_state_dict.pth" \
  --dfn5b-dir "${ASSET_ROOT}/DFN5B-CLIP-ViT-H-14-384" \
  --bigvgan-dir "${ASSET_ROOT}/bigvgan_v2_44khz_128band_512x" \
  --output "${OUTPUT_MODEL}"

echo "Exported FastVideo MMAudio model: ${OUTPUT_MODEL}"
