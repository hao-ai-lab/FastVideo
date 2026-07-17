#!/usr/bin/env bash
set -euo pipefail
CONDITION="${1:?A12/A13/A14/A15}"
RUN_ROOT="${2:?prepared condition run root}"
BASE_PORT="${3:-29800}"
REPO="${REPO:-/mnt/nfs/vlm-k1kong/FastVideo-openvid-a12-a15-final-20260717}"
ENV_DIR="${ENV_DIR:-/mnt/nfs/vlm-k1kong/envs/fastvideo}"
WANDB_MODE="${WANDB_MODE:-online}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
case "$PREFLIGHT_ONLY" in
  0|1) ;;
  *)
    echo "PREFLIGHT_ONLY must be 0 or 1, got: $PREFLIGHT_ONLY" >&2
    exit 2
    ;;
esac
case "$WANDB_MODE" in
  online)
    if [[ "$PREFLIGHT_ONLY" == 0 && -z "${WANDB_API_KEY:-}" ]]; then
      echo "WANDB_MODE=online requires WANDB_API_KEY at runtime." >&2
      exit 2
    fi
    ;;
  offline)
    unset WANDB_API_KEY
    ;;
  *)
    echo "Unsupported WANDB_MODE=$WANDB_MODE; expected online or offline." >&2
    exit 2
    ;;
esac
export REPO ENV_DIR CONDITION RUN_ROOT WANDB_MODE PREFLIGHT_ONLY

run_stage() {
  local stage="$1" final="$2" port="$3"
  if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
    STAGE="$stage" MASTER_PORT="$port" bash "$RUN_ROOT/scripts/train_stage.sh"
    return
  fi
  if [[ -d "$RUN_ROOT/$stage/checkpoints/checkpoint-$final/dcp" ]] &&
     [[ "$(cat "$RUN_ROOT/$stage/state/exit_code" 2>/dev/null || true)" == 0 ]]; then
    echo "$CONDITION $stage checkpoint-$final already complete; skipping"
    return
  fi
  STAGE="$stage" MASTER_PORT="$port" bash "$RUN_ROOT/scripts/train_stage.sh"
  test -d "$RUN_ROOT/$stage/checkpoints/checkpoint-$final/dcp"
}
export_stage() {
  local stage="$1" final="$2" role="${3:-student}"
  local checkpoint="$RUN_ROOT/$stage/checkpoints/checkpoint-$final"
  local marker="$RUN_ROOT/export/$stage/.source_checkpoint_fingerprint"
  local checkpoint_hash config_hash git_head current
  checkpoint_hash="$(find "$checkpoint" -type f -printf '%P:%s:%T@\n' | LC_ALL=C sort | sha256sum | awk '{print $1}')"
  config_hash="$(sha256sum "$RUN_ROOT/$stage/config/run.yaml" | awk '{print $1}')"
  git_head="$(git -C "$REPO" rev-parse HEAD)"
  current="$(printf 'git_head=%s\nrole=%s\nconfig_sha256=%s\ncheckpoint_metadata=%s\n' \
    "$git_head" "$role" "$config_hash" "$checkpoint_hash" | sha256sum | awk '{print $1}')"
  if [[ -s "$RUN_ROOT/export/$stage/transformer/model.safetensors" ]] &&
     [[ "$(cat "$marker" 2>/dev/null || true)" == "$current" ]]; then
    echo "$CONDITION $stage export already complete; skipping"
    return
  fi
  "$ENV_DIR/bin/python" -m fastvideo.train.entrypoint.dcp_to_diffusers \
    --role "$role" --config "$RUN_ROOT/$stage/config/run.yaml" \
    --checkpoint "$RUN_ROOT/$stage/checkpoints/checkpoint-$final" \
    --output-dir "$RUN_ROOT/export/$stage" --overwrite \
    2>&1 | tee "$RUN_ROOT/$stage/logs/export.log"
  printf '%s\n' "$current" > "$marker"
}

if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  run_stage tf 3000 "$((BASE_PORT + 1))"
  run_stage cd 2000 "$((BASE_PORT + 2))"
  run_stage sf 1000 "$((BASE_PORT + 3))"
  echo "$CONDITION launcher preflight passed for tf, cd, and sf; no training was started."
  exit 0
fi

mkdir -p "$RUN_ROOT/state"
on_exit() {
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    local current_status
    current_status="$(cat "$RUN_ROOT/state/status" 2>/dev/null || true)"
    if [[ "$current_status" != failed* ]]; then
      printf 'failed\n' > "$RUN_ROOT/state/status"
    fi
    date -Is > "$RUN_ROOT/state/finished_at"
  fi
}
trap on_exit EXIT

printf 'running\n' > "$RUN_ROOT/state/status"; date -Is > "$RUN_ROOT/state/started_at"
run_stage tf 3000 "$((BASE_PORT + 1))"; export_stage tf 3000 student
run_stage cd 2000 "$((BASE_PORT + 2))"; export_stage cd 2000 ema
run_stage sf 1000 "$((BASE_PORT + 3))"; export_stage sf 1000 student_ema
ema_hash="$(sha256sum "$RUN_ROOT/sf/checkpoints/checkpoint-1000/ema/student.safetensors" | awk '{print $1}')"
export_hash="$(sha256sum "$RUN_ROOT/export/sf/transformer/model.safetensors" | awk '{print $1}')"
printf 'checkpoint_ema %s\nexported_ema %s\n' "$ema_hash" "$export_hash" \
  > "$RUN_ROOT/state/ema_sha256.txt"
if [[ "$ema_hash" != "$export_hash" ]]; then
  printf 'failed_ema_hash_mismatch\n' > "$RUN_ROOT/state/status"
  exit 1
fi
printf 'completed\n' > "$RUN_ROOT/state/status"; date -Is > "$RUN_ROOT/state/finished_at"
