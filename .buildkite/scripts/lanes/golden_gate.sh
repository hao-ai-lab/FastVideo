#!/usr/bin/env bash
# Canonical Slurm CI selection for the golden-gate lane. Environment (HF_HOME
# and authentication) is the runner's responsibility.
set -euo pipefail

exec pytest ./fastvideo/tests/golden_gate -vs
