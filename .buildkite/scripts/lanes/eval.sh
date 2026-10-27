#!/usr/bin/env bash
# Canonical Slurm CI selection for the evaluation lane.
set -euo pipefail

exec pytest ./fastvideo/tests/eval -vs
