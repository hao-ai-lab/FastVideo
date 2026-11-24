#!/usr/bin/env bash
# Canonical Slurm CI selection for the LoRA-inference lane.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../spark_kernel_sync.sh"

exec pytest ./fastvideo/tests/inference/lora/test_lora_inference_similarity.py -vs
