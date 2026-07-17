#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:?prepared experiment root}"
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
export WANDB_MODE PREFLIGHT_ONLY
for condition in A12 A13 A14 A15; do
  bash "$ROOT/scripts/run_condition.sh" "$condition" "$ROOT/$condition" "$((29800 + 10#${condition#A} * 10))"
done
