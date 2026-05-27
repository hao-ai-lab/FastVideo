#!/usr/bin/env bash
# Launch inference from a nested config.
#
# Usage:
#   bash scripts/inference/run.sh <config.yaml> [--dotted.key value ...]
#
# Examples:
#   bash scripts/inference/run.sh scripts/inference/inference_wan.yaml
#   bash scripts/inference/run.sh scripts/inference/inference_wan.yaml \
#       --request.sampling.seed 42 \
#       --generator.engine.num_gpus 2

set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml> [--dotted.key value ...]}"
shift

echo "=== FastVideo Inference ==="
echo "Config: ${CONFIG}"
echo "Extra args: $*"
echo "==========================="

fastvideo generate --config "${CONFIG}" "$@"
