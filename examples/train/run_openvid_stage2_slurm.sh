#!/usr/bin/env bash
# Launch a stage-2 finetuning run (MotionStream-style track masking, lr 1e-6,
# 800 steps) on a HELD node allocation, like ``run_openvid_bidir_held.sh``.
#
# Two variants (pass VARIANT=frozen or VARIANT=unfrozen):
#   VARIANT=frozen    → freezes track_encoder (matches MotionStream paper)
#   VARIANT=unfrozen  → keeps track_encoder trainable (ablation)
#
# WANDB_API_KEY must be exported before calling.
#
# Usage:
#   WANDB_API_KEY=<key> VARIANT=frozen bash examples/train/run_openvid_stage2_slurm.sh [num_nodes]
set -euo pipefail
VARIANT="${VARIANT:-frozen}"
NODES="${1:-4}"; shift || true
: "${WANDB_API_KEY:?export WANDB_API_KEY before launching}"

case "$VARIANT" in
  frozen)
    CFG=examples/train/scenario/worldmodel/finetune_wantrack_openvid_stage2_frozen.yaml
    JOB=openvid_stage2_frozen
    PORT=30100
    FREEZE=1
    ;;
  unfrozen)
    CFG=examples/train/scenario/worldmodel/finetune_wantrack_openvid_stage2_unfrozen.yaml
    JOB=openvid_stage2_unfrozen
    PORT=30200
    FREEZE=0
    ;;
  *) echo "unknown VARIANT=$VARIANT (frozen|unfrozen)"; exit 1 ;;
esac

# Stage-2 recipe: stochastic mid-frame track masking (MotionStream Sec. 3.1)
export WANTRACK_AUG=1
export WANTRACK_SPARSE=1
export WANTRACK_EXTRA_RANDOM=20
export WANTRACK_EXTRA_MODE=random
export WANTRACK_PMASK=0.2               # 20% chance to zero contiguous frame chunks
export WANTRACK_MASK_CHUNK=8            # 8-frame chunks (~1/3 sec at 24 fps)
export WANTRACK_FIXED_SAMPLE=0
export WANTRACK_MOTION_DROP=0
export WANTRACK_TEXT_DROP=0
export WANTRACK_DEBUG="${WANTRACK_DEBUG:-1}"
export WANTRACK_FREEZE_HEAD="$FREEZE"   # 1=freeze track_encoder (MotionStream), 0=trainable
export TRACKWAN_TRACK_BIAS=1            # merged-bias init used bias=True convs

export WANDB_MODE=online
export CFG JOB PORT

# call the same held-alloc launcher used for stage 1
CFG="$CFG" JOB="$JOB" PORT="$PORT" \
  bash "$(dirname "$0")/run_openvid_bidir_held.sh" "$NODES" "$@"
