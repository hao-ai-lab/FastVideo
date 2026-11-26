#!/usr/bin/env bash
# Canonical Slurm CI selection for the LoRA-extraction lane.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../spark_kernel_sync.sh"

exec pytest ./fastvideo/tests/lora_extraction/ -vs
