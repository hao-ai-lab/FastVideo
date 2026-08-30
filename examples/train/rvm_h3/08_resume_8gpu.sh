#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
activate_rvm_env

CONFIG="${1:?Usage: $0 <config.yaml> <checkpoint-dir|latest> [extra overrides...]}"
CHECKPOINT="${2:?Usage: $0 <config.yaml> <checkpoint-dir|latest> [extra overrides...]}"
shift 2

export NUM_GPUS="${NUM_GPUS:-8}"
export RVM_SP_SIZE="${RVM_SP_SIZE:-4}"
run_rvm_training \
    "${CONFIG}" \
    --training.checkpoint.resume_from_checkpoint "${CHECKPOINT}" \
    "$@"

cat <<'EOF'
RVM endpoints are intentionally not checkpointed. If a job dies between the
two optimizer updates attached to one rollout collection, resume regenerates
that collection from the restored RNG/dataloader state rather than serializing
hundreds of decoded/latent samples. Compare the first resumed validation and
reward statistics before continuing a long run.
EOF
