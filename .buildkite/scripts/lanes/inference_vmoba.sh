#!/usr/bin/env bash
# Canonical Slurm CI selection for the VMoBA-inference lane.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../spark_kernel_sync.sh"

exec pytest ./fastvideo/tests/inference/vmoba/test_vmoba_inference.py -vs
