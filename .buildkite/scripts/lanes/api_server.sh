#!/usr/bin/env bash
# Canonical Slurm CI selection for the OpenAI-compatible API lane.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../spark_kernel_sync.sh"

exec pytest ./fastvideo/tests/entrypoints/test_openai_api_integration.py -vs
