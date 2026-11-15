#!/usr/bin/env bash
# Canonical Slurm CI selection for the modular training-framework lane.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../spark_kernel_sync.sh"

exec pytest ./fastvideo/tests/train/models ./fastvideo/tests/train/methods -vs
