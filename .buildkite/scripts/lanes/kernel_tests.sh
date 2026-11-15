#!/usr/bin/env bash
# Canonical Slurm CI selection for the custom-kernel lane.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../spark_kernel_sync.sh"

exec pytest fastvideo-kernel/tests/ -vs
