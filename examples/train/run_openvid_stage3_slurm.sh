#!/usr/bin/env bash
# Stage-3 experiments: per-track dropout (WANTRACK_TRACK_DROP=0.5) to fix the
# sparse-conditioning distribution mismatch at inference. Three variants:
#
#   VARIANT=A_chunkmask     TRACK_DROP=0.5 + PMASK=0.2       (from stage-2 frozen 800)
#   VARIANT=B_motiondrop    TRACK_DROP=0.5 + MOTION_DROP=0.10 (from stage-2 frozen 800)
#   VARIANT=C_from3700      TRACK_DROP=0.5 + PMASK=0.2       (from merged-bias 3700 — earlier ckpt sanity)
#
# All three: lr 1e-6, freeze head, 600 steps.
set -euo pipefail
VARIANT="${VARIANT:-A_chunkmask}"
NODES="${1:-4}"; shift || true
: "${WANDB_API_KEY:?export WANDB_API_KEY before launching}"

case "$VARIANT" in
  A_chunkmask)
    CFG=examples/train/scenario/worldmodel/finetune_wantrack_openvid_stage3A_chunkmask.yaml
    JOB=openvid_stage3A_chunkmask
    PORT=30300
    PMASK=0.2; MOTION_DROP=0.0
    ;;
  B_motiondrop)
    CFG=examples/train/scenario/worldmodel/finetune_wantrack_openvid_stage3B_motiondrop.yaml
    JOB=openvid_stage3B_motiondrop
    PORT=30400
    PMASK=0.0; MOTION_DROP=0.10
    ;;
  C_from3700)
    CFG=examples/train/scenario/worldmodel/finetune_wantrack_openvid_stage3C_from3700.yaml
    JOB=openvid_stage3C_from3700
    PORT=30500
    PMASK=0.2; MOTION_DROP=0.0
    ;;
  *) echo "unknown VARIANT=$VARIANT"; exit 1 ;;
esac

export WANTRACK_AUG=1
export WANTRACK_SPARSE=1
export WANTRACK_EXTRA_RANDOM=20
export WANTRACK_EXTRA_MODE=random
export WANTRACK_TRACK_DROP=0.5              # per-track dropout (the fix)
export WANTRACK_PMASK="$PMASK"              # chunked mid-frame mask (0 for B)
export WANTRACK_MASK_CHUNK=8
export WANTRACK_MOTION_DROP="$MOTION_DROP"  # full-track drop (0 for A/C)
export WANTRACK_TEXT_DROP=0
export WANTRACK_FIXED_SAMPLE=0
export WANTRACK_DEBUG="${WANTRACK_DEBUG:-1}"
export WANTRACK_FREEZE_HEAD=1               # head is converged
export TRACKWAN_TRACK_BIAS=1

export WANDB_MODE=online
export CFG JOB PORT

CFG="$CFG" JOB="$JOB" PORT="$PORT" \
  bash "$(dirname "$0")/run_openvid_bidir_held.sh" "$NODES" "$@"
