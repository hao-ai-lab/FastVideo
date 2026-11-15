#!/usr/bin/env bash
# Canonical Slurm CI selection for the VAE lane.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../spark_kernel_sync.sh"

pytest ./fastvideo/tests/golden_gate/test_wan_vae.py -xvs
exec pytest ./fastvideo/tests/vaes -vs
