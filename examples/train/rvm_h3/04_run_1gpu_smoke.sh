#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
activate_rvm_env
require_path "${FASTH3_MODEL_DIR}"
require_path "${RVM_SMOKE_DATA}"
require_path "${RVM_EVAL_DATA}"

export NUM_GPUS=1
export RVM_SP_SIZE=1
run_rvm_training \
    examples/train/configs/rl/minimax_h3/rvm_h3_1gpu_smoke.yaml \
    --training.checkpoint.output_dir "${RVM_1GPU_OUTPUT:-outputs/rvm_h3/1gpu_smoke}" \
    --training.tracker.run_name "${RVM_1GPU_RUN_NAME:-rvm-h3-1gpu-smoke}"

cat <<'EOF'
The smoke is accepted only after checking:
  1. baseline validation rendered at step 0;
  2. four optimizer steps are finite;
  3. reward components are finite and group std is nonzero;
  4. videos remain dynamic and audio remains present;
  5. checkpoint save/resume and LoRA export work.
EOF
