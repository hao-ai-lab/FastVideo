#!/bin/bash
# usage: score_all.sh DIR...
REF=/mnt/nfs/vlm-aryan/regen2k/out/parity_fox/ref_frames_u8.pt
source /mnt/nfs/vlm-aryan/regen2k-fast/scripts/common.sh
for d in "$@"; do
  echo "== $d"
  $PY $W/scripts/score_run.py --ref $REF --dir $d
done
