#!/usr/bin/env bash
# Seed step B's output dir with step A's final checkpoint, so B starts from A while its config
# can still use resume_from_checkpoint: latest (which is what makes crash-restarts safe).
#
# Hardlinked (cp -al), not copied: a 14B training-state checkpoint is hundreds of GB (bf16
# weights + fp32 master + Adam m/v), and both dirs live on the same Lustre filesystem.
# Checkpoints are write-once, so sharing inodes is safe here.
set -euo pipefail
WORK=/mnt/lustre/vlm-s4duan
STEP="${STEP:-1000}"
SRC="${SRC:-$WORK/wantrack_14b_synth_sparse_fixed_out/checkpoint-$STEP}"
DST_DIR="${DST_DIR:-$WORK/wantrack_14b_synth_sparse_random_out}"
DST="$DST_DIR/checkpoint-$STEP"

[ -d "$SRC" ] || { echo "[B-seed] missing $SRC"; exit 1; }
[ -f "$SRC/dcp/.metadata" ] || { echo "[B-seed] $SRC incomplete (no dcp/.metadata) — refusing"; exit 1; }

if [ -d "$DST" ]; then
  echo "[B-seed] $DST already exists — leaving it alone"
else
  mkdir -p "$DST_DIR"
  echo "[B-seed] hardlinking $SRC -> $DST"
  cp -al "$SRC" "$DST"
fi
echo "[B-seed] checkpoints now in $DST_DIR:"
ls -d "$DST_DIR"/checkpoint-* 2>/dev/null | sed 's/^/  /'
