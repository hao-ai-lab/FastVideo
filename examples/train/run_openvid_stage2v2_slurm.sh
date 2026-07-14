#!/usr/bin/env bash
# Stage-2 REPLACEMENT (hi-LR): the empirical eval showed stages 2/3 at lr=1e-6
# barely moved weights (~0.02%/600 steps), so behavior barely changed. Try lr=5e-5
# to actually shift the model. Aggressive drop rates so it truly sees the sparse
# regime the user hits at inference time.
#
# Init: merged_bias_ckpt4800 (stage-1 end, clean base).
# Recipe: TRACK_DROP=0.5 + MOTION_DROP=0.3 + PMASK=0.2 + freeze head.
set -euo pipefail
NODES="${1:-4}"; shift || true
: "${WANDB_API_KEY:?export WANDB_API_KEY before launching}"

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

CFG=examples/train/scenario/worldmodel/finetune_wantrack_openvid_stage2v2_hilr.yaml
JOB=openvid_stage2v2_hilr
PORT=30600

CFG="$CFG" JOB="$JOB" PORT="$PORT" \
  bash "$(dirname "$0")/run_openvid_bidir_held.sh" "$NODES" "$@"
