#!/usr/bin/env bash
set -euo pipefail
CONDITION="${1:?A12/A13/A14/A15}"
RUN_ROOT="${2:?prepared condition run root}"
BASE_PORT="${3:-29800}"
REPO="${REPO:-/mnt/nfs/vlm-k1kong/FastVideo-openvid-a12-a15-plan-20260717}"
ENV_DIR="${ENV_DIR:-/mnt/nfs/vlm-k1kong/envs/fastvideo}"
: "${WANDB_API_KEY:?export WANDB_API_KEY at launch time}"
export REPO ENV_DIR CONDITION RUN_ROOT

run_stage() {
  local stage="$1" final="$2" port="$3"
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
  if [[ -s "$RUN_ROOT/export/$stage/transformer/model.safetensors" ]]; then
    echo "$CONDITION $stage export already complete; skipping"
    return
  fi
  "$ENV_DIR/bin/python" -m fastvideo.train.entrypoint.dcp_to_diffusers \
    --role "$role" --config "$RUN_ROOT/$stage/config/run.yaml" \
    --checkpoint "$RUN_ROOT/$stage/checkpoints/checkpoint-$final" \
    --output-dir "$RUN_ROOT/export/$stage" --overwrite \
    2>&1 | tee "$RUN_ROOT/$stage/logs/export.log"
}

printf 'running\n' > "$RUN_ROOT/state/status"; date -Is > "$RUN_ROOT/state/started_at"
run_stage tf 3000 "$((BASE_PORT + 1))"; export_stage tf 3000 student
run_stage cd 2000 "$((BASE_PORT + 2))"; export_stage cd 2000 student
run_stage sf 1000 "$((BASE_PORT + 3))"; export_stage sf 1000 student_ema
sha256sum "$RUN_ROOT/sf/checkpoints/checkpoint-1000/ema/student.safetensors" \
  "$RUN_ROOT/export/sf/transformer/model.safetensors" > "$RUN_ROOT/state/ema_sha256.txt"
printf 'completed\n' > "$RUN_ROOT/state/status"; date -Is > "$RUN_ROOT/state/finished_at"
