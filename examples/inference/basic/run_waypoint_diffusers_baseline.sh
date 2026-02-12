#!/usr/bin/env bash
# Run Waypoint diffusers baseline with cache/tmp on /workspace (avoids RunPod disk quota).
# Use this on RunPod so env is set before Python starts:
#   ./examples/inference/basic/run_waypoint_diffusers_baseline.sh
# Or: bash examples/inference/basic/run_waypoint_diffusers_baseline.sh
set -e
if [[ -d /workspace ]]; then
  export HF_HOME=/workspace/.cache/huggingface
  export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
  export HF_HUB_CACHE=/workspace/.cache/huggingface/hub
  export HF_XET_CACHE=/workspace/.cache/huggingface/xet
  export TMPDIR=/workspace/tmp
  export HF_HUB_DISABLE_XET=1
  mkdir -p /workspace/.cache/huggingface/hub /workspace/.cache/huggingface/xet /workspace/tmp
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROOT_DIR"
exec python "$SCRIPT_DIR/basic_waypoint_diffusers_baseline.py" "$@"
