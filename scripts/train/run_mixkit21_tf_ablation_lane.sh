#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/mnt/nfs/vlm-k1kong/FastVideo-causal-kernel}"
EXP_ROOT="${EXP_ROOT:-/mnt/lustre/vlm-k1kong/experiments/mixkit21_tf2k_kernel_ablation}"
SEQUENCE_ID="${SEQUENCE_ID:?Set a shared timestamped SEQUENCE_ID}"
LANE="${LANE:?Set LANE to node0 or node1}"
RUN_CONDITIONS="${RUN_CONDITIONS:?Set the ordered experiment IDs for this lane}"
PROJECT_NAME="${PROJECT_NAME:-causal_forcing_mixkit21_kernel_tf2k_long249_ablation}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29800}"
MATRIX="${MATRIX:-examples/train/configs/ablation/wan_causal_mixkit21/experiment_matrix.tsv}"

SEQUENCE_DIR="$EXP_ROOT/sequences/$SEQUENCE_ID"
LANE_LOG="$SEQUENCE_DIR/${LANE}.log"
LANE_TABLE="$SEQUENCE_DIR/${LANE}_runs.tsv"
mkdir -p "$SEQUENCE_DIR"
printf '%s\n' "$SEQUENCE_DIR" > "$EXP_ROOT/FULL_SEQUENCE_LATEST"
printf 'id\tcondition\trun_root\tstatus_file\n' > "$LANE_TABLE"

log() {
  printf '[%s] [%s] %s\n' "$(date -Is)" "$LANE" "$*" | tee -a "$LANE_LOG"
}

lookup_condition() {
  awk -F '\t' -v id="$1" 'NR > 1 && $1 == id { print $2 }' "$REPO/$MATRIX"
}

log "Starting conditions: $RUN_CONDITIONS"
index=0
for experiment_id in $RUN_CONDITIONS; do
  condition="$(lookup_condition "$experiment_id")"
  if [[ -z "$condition" ]]; then
    log "Unknown experiment ID: $experiment_id"
    exit 2
  fi
  run_root="$EXP_ROOT/runs/${SEQUENCE_ID}_${experiment_id}_${condition}_tf2k_mixkit21_val249_triton"
  status_file="$run_root/tf/state/status"
  printf '%s\t%s\t%s\t%s\n' "$experiment_id" "$condition" "$run_root" "$status_file" >> "$LANE_TABLE"
  port=$((MASTER_PORT_BASE + index))
  log "$experiment_id start: $run_root"
  CONDITION="$experiment_id" \
  RUN_ROOT="$run_root" \
  MASTER_PORT="$port" \
  PROJECT_NAME="$PROJECT_NAME" \
  WANDB_RUN_NAME="tf2k_${experiment_id}_${condition}_${SEQUENCE_ID}" \
  bash "$REPO/scripts/train/train_mixkit21_tf_ablation.sh"
  log "$experiment_id complete"
  index=$((index + 1))
done
log "Lane complete"
