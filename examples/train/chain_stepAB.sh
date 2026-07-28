#!/usr/bin/env bash
# Wait for step A to finish, then seed and launch step B — so the held allocation never sits
# idle between the two overfit stages.
#
# "A is done" means BOTH: no step-A launcher process is alive, AND a COMPLETE checkpoint-1000
# exists (dcp/.metadata present). Requiring both avoids launching B off a half-written
# checkpoint if A died mid-save.
set -uo pipefail
WORK=/mnt/lustre/vlm-s4duan
REPO=$WORK/FastVideo
ALLOC="${ALLOC:-728}"
A_JOB="${A_JOB:-wan14b_stepA_v2}"
A_CKPT="$WORK/wantrack_14b_synth_sparse_fixed_out/checkpoint-1000"
: "${WANDB_API_KEY:?export WANDB_API_KEY}"
say() { echo "[chain $(date +%H:%M:%S)] $*"; }

say "waiting for step A to reach a complete $A_CKPT ..."
while :; do
  if [ -f "$A_CKPT/dcp/.metadata" ]; then
    # Checkpoint is complete; wait for the launcher to actually exit before reusing the nodes.
    # Use the launcher's PID file, NOT `pgrep -f run_wan14b_held.sh` — that pattern also matches
    # any watcher/shell whose command line contains the string, so the check never goes false
    # and the chain waits forever on nodes that are already idle.
    A_RUNFILE="$WORK/logs/${A_JOB}.running"
    if [ ! -f "$A_RUNFILE" ] || ! kill -0 "$(cat "$A_RUNFILE" 2>/dev/null)" 2>/dev/null; then
      say "step A finished and $A_CKPT is complete"
      break
    fi
    say "checkpoint-1000 complete, waiting for step A launcher (pid $(cat "$A_RUNFILE")) to exit ..."
  fi
  [ -n "$(squeue -h -j "$ALLOC" -o '%t' 2>/dev/null)" ] || { say "ALLOC $ALLOC vanished — aborting chain"; exit 1; }
  sleep 60
done

say "seeding step B output dir from A's checkpoint-1000"
bash "$REPO/examples/train/run_stepB_seed.sh" || { say "seed failed"; exit 1; }

say "launching step B (random-track overfit, WANTRACK_FIXED_SAMPLE=0)"
cd "$REPO"
exec env ALLOC="$ALLOC" NODES=4 JOB=wan14b_stepB_v2 PORT=30918 \
  CFG=examples/train/scenario/worldmodel/finetune_wantrack_synth_sparse_random_14b.yaml \
  WANTRACK_FIXED_SAMPLE=0 WANTRACK_FREEZE_HEAD=0 TRACKWAN_TRACK_BIAS=1 \
  WANDB_API_KEY="$WANDB_API_KEY" \
  bash examples/train/run_wan14b_held.sh
