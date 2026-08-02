#!/bin/bash
# Seed Stage B's output dir with Stage A's FINAL checkpoint, so B runs in its own dir while its
# config still uses resume_from_checkpoint: latest (which is what makes crash-restarts safe).
# Mirrors upstream examples/train/run_stepB_seed.sh.
#
# Run this ONCE (not per-rack) on the shared filesystem, AFTER Stage A finishes, BEFORE launching
# Stage B. Hardlinked (cp -al), not copied: a 14B training-state checkpoint is 100s of GB, and both
# dirs are on the same filesystem (/home), so hardlinks are instant and use no extra space.
#
#   bash data_pipeline/720_stage_1/02b_seed_stageB.sh
#   # then, on BOTH racks:
#   MASTER_ADDR=hpc-rack-1-6 NODE_RANK=0 STAGE=B bash data_pipeline/720_stage_1/02_run_overfit.sh \
#     --training.loop.max_train_steps 3000
set -uo pipefail

OUT_A=${OUT_A:-/home/hal-kevin/data/motion-stream-test/overfit_14b_720p_d64_bias_out}
OUT_B=${OUT_B:-/home/hal-kevin/data/motion-stream-test/overfit_14b_720p_d64_bias_stageB_out}

# Pick A's latest COMPLETE checkpoint (dcp/.metadata present) — refuse to seed off a partial save.
STEP=${STEP:-}
if [ -z "$STEP" ]; then
  for d in $(for c in "$OUT_A"/checkpoint-*; do n=${c##*checkpoint-}; echo "$n"; done | sort -rn); do
    [ -f "$OUT_A/checkpoint-$d/dcp/.metadata" ] && { STEP=$d; break; }
  done
fi
[ -n "$STEP" ] || { echo "[B-seed] no complete checkpoint found under $OUT_A" >&2; exit 1; }

SRC="$OUT_A/checkpoint-$STEP"
DST="$OUT_B/checkpoint-$STEP"
[ -f "$SRC/dcp/.metadata" ] || { echo "[B-seed] $SRC incomplete (no dcp/.metadata) — refusing" >&2; exit 1; }

mkdir -p "$OUT_B"
if [ -e "$DST" ]; then
  echo "[B-seed] $DST already exists — leaving it alone"
else
  echo "[B-seed] hardlinking $SRC -> $DST"
  cp -al "$SRC" "$DST"
fi
echo "[B-seed] Stage B dir now seeded at step $STEP:"
for c in "$OUT_B"/checkpoint-*; do echo "  $c"; done
echo "[B-seed] Launch B with --training.loop.max_train_steps > $STEP (e.g. $((STEP + 1000)))."
