#!/usr/bin/env bash
# Synth stage-2: MotionStream-style fine-tuning on 49k Wan2.2 synth data.
# Init: merged_bias_ckpt4800 (our stage-1 end on openvid).
# Recipe: TRACK_DROP=0.5 + MOTION_DROP=0.3 + PMASK=0.2 + freeze head.
# 600 steps -> ~1.55 epochs on combined 49k (paper claims "~1 epoch" for stage-2).
# Two VARIANTs: paperLR (1e-6, paper spec) or 5x (5e-6, 5x paper, 2x below stage-1's 1e-5).
set -euo pipefail
VARIANT="${VARIANT:-paperLR}"
NODES="${1:-4}"; shift || true
: "${WANDB_API_KEY:?export WANDB_API_KEY before launching}"

case "$VARIANT" in
  paperLR)
    CFG=examples/train/scenario/worldmodel/finetune_wantrack_synth_stage2_paperLR.yaml
    JOB=synth_stage2_paperLR
    PORT=30700
    ;;
  5x)
    CFG=examples/train/scenario/worldmodel/finetune_wantrack_synth_stage2_5x.yaml
    JOB=synth_stage2_5x
    PORT=30800
    ;;
  *) echo "unknown VARIANT=$VARIANT"; exit 1 ;;
esac

export WANTRACK_AUG=1
export WANTRACK_SPARSE=1
export WANTRACK_EXTRA_RANDOM=20
export WANTRACK_EXTRA_MODE=random
export WANTRACK_TRACK_DROP=0.5
export WANTRACK_MOTION_DROP=0.3
export WANTRACK_PMASK=0.2
export WANTRACK_MASK_CHUNK=8
export WANTRACK_TEXT_DROP=0
export WANTRACK_FIXED_SAMPLE=0
export WANTRACK_DEBUG="${WANTRACK_DEBUG:-1}"
export WANTRACK_FREEZE_HEAD=1
export TRACKWAN_TRACK_BIAS=1

export WANDB_MODE=online

CFG="$CFG" JOB="$JOB" PORT="$PORT" \
  bash "$(dirname "$0")/run_openvid_bidir_held.sh" "$NODES" "$@"
