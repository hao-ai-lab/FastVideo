#!/bin/bash
# Step 06 — export the OpenVid Stage-1 DCP checkpoint to a diffusers dir. Stage 2 (07) inits its
# WEIGHTS from this export (init_from, fresh optimizer/step), so it must be a diffusers model dir,
# not a DCP resume. Only 1 GPU needed. Run on a compute node.
set -uo pipefail
cd ~/FastVideo

CKPT=${CKPT:-/home/hal-kevin/data/motion-stream-test/openvid_stage1_14b_720p_out}   # auto-picks latest
OUT=${OUT:-/home/hal-kevin/models/openvid_stage1_14b_export}
CFG=${CFG:-data_pipeline/720_stage_1/finetune_wantrack_openvid_stage1_14b_720p_d64_bias.yaml}

python -m fastvideo.train.entrypoint.dcp_to_diffusers \
  --checkpoint "$CKPT" \
  --output-dir "$OUT" \
  --config "$CFG" \
  --role student \
  --overwrite

echo "[export] stage-1 teacher -> $OUT"
echo "[export] next: 07_run_synth_stage2.sh (init_from defaults to this dir)"
