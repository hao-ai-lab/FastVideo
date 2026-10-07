#!/usr/bin/env bash
# Canonical Slurm CI selection for the modular training-framework lane.
set -euo pipefail

exec pytest ./fastvideo/tests/train/models ./fastvideo/tests/train/methods -vs
