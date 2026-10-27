#!/usr/bin/env bash
# Canonical Slurm CI selection for the distillation-DMD lane.
set -euo pipefail

exec pytest ./fastvideo/tests/training/distill/test_distill_dmd.py -vs
