#!/usr/bin/env bash
# Sample the WanTrack gate at each new checkpoint and append to a TSV, so we can see whether it
# is on track for the ~0.011 the successful 1.3B merged init reached. Checkpoints rotate
# (total_limit=3), so poll faster than they are produced or the early trajectory is lost.
set -uo pipefail
WORK=/mnt/lustre/vlm-s4duan; REPO=$WORK/FastVideo
OUT="${OUT:-$WORK/wantrack_14b_synth_sparse_fixed_out}"
LOG="${LOG:-$WORK/logs/gate_growth.tsv}"
cd "$REPO"; source .venv/bin/activate
export HOME=$WORK PYTHONPATH=$REPO NCCL_CUMEM_ENABLE=0 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
       MASTER_ADDR=127.0.0.1 MASTER_PORT=29600
touch "$LOG"
while :; do
  for d in "$OUT"/checkpoint-*; do
    [ -f "$d/dcp/.metadata" ] || continue
    step=$(basename "$d" | sed 's/checkpoint-//')
    grep -q "^${step}	" "$LOG" 2>/dev/null && continue
    line=$(python data_pipeline/check_gate_growth.py "$d" 2>/dev/null | grep -oE "gate\[:, 36:\] std=[0-9.]+" | grep -oE "[0-9.]+$")
    [ -n "$line" ] && { printf "%s\t%s\t%s\n" "$step" "$line" "$(date +%H:%M:%S)" >> "$LOG"; echo "step $step gate=$line"; }
  done
  pgrep -f "run_wan14b_held.sh" >/dev/null || { echo "[watch] no launcher; exiting"; break; }
  sleep 240
done
