#!/usr/bin/env bash
# Profile Wan2.1 T2V 1.3B on one DGX Spark GB10 GPU with Nsight Systems.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/configs/wan-t2v-1.3b-gb10-1gpu.json}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/nsys/gb10}"

exec "${SCRIPT_DIR}/profile_wan_t2v_1_3b_nsys.sh" "$@"
