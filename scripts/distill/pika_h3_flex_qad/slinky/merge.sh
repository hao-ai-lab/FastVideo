#!/bin/bash
# Download the two Pika H3 Flex adapters and merge each into its MiniMax-H3 trunk (runs under env.sh).
set -euo pipefail
R=/mnt/nfs/vlm-aryan/pika-flex-qad
BASE=$HF_HOME/hub/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe
ADAPTERS=$R/models/pika-h3-flex-adapters
for task in ref2va fl2va; do
  out=$R/models/pika-flex-$task-merged
  if [ -f $out/pika_flex_qad.json ]; then echo "$out exists, skipping"; continue; fi
  rm -rf $out
  time $PY scripts/distill/pika_h3_flex_qad/merge_pika_adapter.py --base $BASE --adapter $ADAPTERS --task $task --output $out
  cat $out/pika_flex_qad.json
done
