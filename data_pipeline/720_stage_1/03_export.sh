#!/bin/bash
# Phase 0, step 3 — export the overfit DCP checkpoint to a diffusers model dir.
# The Phase-1 merge (convert_trackwan_init_v2.py --track-src/--pe-src) reads this diffusers dir.
# Only 1 GPU needed (DCP reshards automatically). Run on a compute node.
set -uo pipefail
cd ~/FastVideo

# --checkpoint accepts an output_dir (auto-picks the latest checkpoint-<step>), a specific
# checkpoint-<step> dir, or its dcp/ subdir.
# Default = the Stage-B dir (the FINAL overfit; B refines A and is what the merge lifts). Override
# CKPT to the Stage-A dir only if you deliberately want to export A.
CKPT=${CKPT:-/home/hal-kevin/data/motion-stream-test/overfit_14b_720p_d64_bias_stageB_out}
OUT=${OUT:-/home/hal-kevin/models/overfit14b_export}
CFG=${CFG:-data_pipeline/720_stage_1/finetune_wantrack_overfit_14b_720p_d64_bias.yaml}

# TRACKWAN_TRACK_BIAS=1 MUST match training: it toggles bias on track_encoder.{proj,temporal_conv}
# at build time (track_encoder.py:73). The overfit trained with bias=1, so the checkpoint carries
# those bias params; building bias-less here fails to load them ("track_encoder.proj.bias not found").
TRACKWAN_TRACK_BIAS=1 \
python -m fastvideo.train.entrypoint.dcp_to_diffusers \
  --checkpoint "$CKPT" \
  --output-dir "$OUT" \
  --config "$CFG" \
  --role student \
  --overwrite \
  || { echo "[export] FAILED (see traceback above) -- nothing written" >&2; exit 1; }

# Guard against a silent partial write (the entrypoint can exit 0 yet write no weights).
ls "$OUT"/transformer/*.safetensors >/dev/null 2>&1 \
  || { echo "[export] ERROR: no transformer/*.safetensors under $OUT -- export did not complete" >&2; exit 1; }

echo "[export] wrote diffusers model -> $OUT"
echo "[export] next (Phase 1 merge):"
echo "  python data_pipeline/convert_trackwan_init_v2.py \\"
echo "    --base /home/hal-kevin/models/Wan2.1-I2V-14B-720P-Diffusers \\"
echo "    --out  /home/hal-kevin/models/trackwan_14b_i2v_d64_merged_from_overfit_bias \\"
echo "    --id-dim 64 --pe-init random \\"
echo "    --track-src $OUT/transformer --pe-src $OUT/transformer"
