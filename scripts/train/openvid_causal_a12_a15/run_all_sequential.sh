#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:?prepared experiment root}"
: "${WANDB_API_KEY:?export WANDB_API_KEY at launch time}"
for condition in A12 A13 A14 A15; do
  bash "$ROOT/scripts/run_condition.sh" "$condition" "$ROOT/$condition" "$((29800 + 10#${condition#A} * 10))"
done
